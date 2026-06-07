from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


THIS_FILE = Path(__file__).resolve()
WORKSPACE = THIS_FILE.parents[2]
DEFAULT_OUTPUT_DIR = THIS_FILE.parent / "sample_folders"

FULL_KEY = "ns_noise_5000_161_64_64"
FOAE_LATENT_KEY = "latent_DPN_d150_fno"
OAE_LATENT_KEY = "latent_DPN_d150"
AE_LATENT_KEY = "u_latent_fno_d150_m12_w1"
FOAE_TRUNK_KEY = "trunk_DPN_d150_fno"
OAE_TRUNK_KEY = "trunk_DPN_d150"

FOAE_REDUCTION_MODEL = "DPN_d150_fno.pth"
OAE_REDUCTION_MODEL = "DPN_d150.pth"
AE_REDUCTION_MODEL = "fno_d150_m12_w1"

FOAE_TEMPORAL_MODEL = "ns2d_N4900_p18_m76_1e0pi_DPN_d150_fno"
OAE_TEMPORAL_MODEL = "ns2d_N4900_p18_m76_1e0pi_DPN_d150"
AE_TEMPORAL_MODEL = "ns2d_N4900_p18_5e-1pi_fno_d150_m12_w1"

CMAP_NAME = "RdBu"  # low values red, high values blue, white near the middle
IMAGE_SCALE = 8  # 64x64 fields become 512x512 images while preserving the 2D field aspect.
SEED = 20260531


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def compl_mul2d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        usable_1 = min(self.modes1, x_ft.shape[-2])
        usable_2 = min(self.modes2, x_ft.shape[-1])
        out_ft[:, :, :usable_1, :usable_2] = self.compl_mul2d(
            x_ft[:, :, :usable_1, :usable_2],
            self.weights1[:, :, :usable_1, :usable_2],
        )
        out_ft[:, :, -usable_1:, :usable_2] = self.compl_mul2d(
            x_ft[:, :, -usable_1:, :usable_2],
            self.weights2[:, :, :usable_1, :usable_2],
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


# The class name intentionally stays FNO because the latent predictors are commonly
# saved with torch.save(model) from latent.py, where the class is named FNO.
class FNO(nn.Module):
    def __init__(self, modes1: int, modes2: int, width: int) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.conv0 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc0 = nn.Linear(3, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

    @staticmethod
    def get_grid(shape: torch.Size, device: torch.device) -> torch.Tensor:
        batch_size, size_x, size_y = shape[0], shape[1], shape[2]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_x = grid_x.reshape(1, size_x, 1, 1).repeat(batch_size, 1, size_y, 1)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_y = grid_y.reshape(1, 1, size_y, 1).repeat(batch_size, size_x, 1, 1)
        return torch.cat((grid_x, grid_y), dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export NS2D reconstruction and latent-prediction sample folders for F-OAE/OAE/AE."
    )
    parser.add_argument("--dm-root", type=Path, default=None, help="Path to DM-2dns. Auto-detected by default.")
    parser.add_argument("--ae-root", type=Path, default=None, help="Path to AE-2dns. Auto-detected by default.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--image-scale", type=int, default=IMAGE_SCALE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--heldout-start", type=int, default=4900)
    parser.add_argument("--heldout-stop", type=int, default=5000)
    parser.add_argument("--latent-case", type=int, default=None)
    return parser.parse_args()


def autodetect_root(name: str, provided: Path | None) -> Path:
    if provided is not None:
        root = provided.resolve()
        if not root.exists():
            raise FileNotFoundError(f"{name} root does not exist: {root}")
        return root
    candidates = [
        WORKSPACE / name,
        WORKSPACE / "DM-2d" / name,
        THIS_FILE.parent / name,
        THIS_FILE.parent.parent / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find {name}. Pass --{'dm' if name.startswith('DM') else 'ae'}-root explicitly."
    )


def resolve_existing(base: Path, name: str) -> Path:
    candidates = [base / name]
    if not name.endswith(".pth"):
        candidates.append(base / f"{name}.pth")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Return the first candidate so the final FileNotFoundError names the intended path.
    return candidates[0]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def choose_scale(values: list[float]) -> int:
    max_abs = max(abs(value) for value in values)
    if max_abs == 0:
        return 0
    exponent = int(math.floor(math.log10(max_abs)))
    return 0 if 1 <= max_abs < 100 else exponent


def rounded_limits(
    vmin: float, vmax: float, force_zero_bottom: bool = False
) -> tuple[float, float, float, int, tuple[float, float, float]]:
    exponent = choose_scale([vmin, vmax])
    scale = 10.0**exponent
    low_i = 0 if force_zero_bottom else math.floor(vmin / scale * 10.0)
    high_i = math.ceil(vmax / scale * 10.0)
    if high_i == low_i:
        high_i += 1
    if (low_i + high_i) % 2 != 0:
        if force_zero_bottom or abs(high_i) <= abs(low_i):
            high_i += 1
        else:
            low_i -= 1
    low = low_i / 10.0 * scale
    high = high_i / 10.0 * scale
    mid = 0.5 * (low + high)
    return low, high, mid, exponent, (high / scale, mid / scale, low / scale)


def save_field_image(array: np.ndarray, path: Path, vmin: float, vmax: float, image_scale: int) -> None:
    field = np.asarray(array, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError(f"Expected a 2D field, got {field.shape}.")
    rgba = plt.get_cmap(CMAP_NAME)(colors.Normalize(vmin=vmin, vmax=vmax)(field), bytes=True)
    height, width = field.shape
    image = Image.fromarray(rgba).resize((width * image_scale, height * image_scale), Image.Resampling.BICUBIC)
    image.save(path)


def save_colorbar(
    path: Path,
    vmin: float,
    vmax: float,
    mid: float,
    exponent: int,
    labels: tuple[float, float, float],
) -> None:
    fig = plt.figure(figsize=(2.75, 1.12), dpi=300)
    cax = fig.add_axes([0.12, 0.72, 0.76, 0.15])
    sm = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=CMAP_NAME)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([vmin, mid, vmax])
    cb.set_ticklabels([f"{labels[2]:.1f}", f"{labels[1]:.1f}", f"{labels[0]:.1f}"])
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.tick_params(labelsize=12, length=0, pad=3)
    cb.outline.set_linewidth(0.8)
    if exponent != 0:
        fig.text(0.64, 0.62, rf"$\times 10^{{{exponent}}}$", fontsize=10, ha="left", va="center")
    fig.savefig(path, transparent=False)
    plt.close(fig)


def mse(prediction: np.ndarray, benchmark: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - benchmark)))


def clean_pngs(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()


def save_sample_outputs(
    folder: Path,
    benchmark: np.ndarray,
    foae: np.ndarray,
    oae: np.ndarray,
    ae: np.ndarray,
    image_scale: int,
) -> dict[str, float]:
    clean_pngs(folder)
    fields = [benchmark, foae, oae, ae]
    errors = [np.abs(foae - benchmark), np.abs(oae - benchmark), np.abs(ae - benchmark)]
    field_limits = rounded_limits(float(min(field.min() for field in fields)), float(max(field.max() for field in fields)))
    error_limits = rounded_limits(0.0, float(max(error.max() for error in errors)), force_zero_bottom=True)
    values = {"foae": mse(foae, benchmark), "oae": mse(oae, benchmark), "ae": mse(ae, benchmark)}

    save_field_image(benchmark, folder / "01_benchmark.png", field_limits[0], field_limits[1], image_scale)
    save_field_image(foae, folder / "02_F-OAE.png", field_limits[0], field_limits[1], image_scale)
    save_field_image(oae, folder / "03_OAE.png", field_limits[0], field_limits[1], image_scale)
    save_field_image(ae, folder / "04_AE.png", field_limits[0], field_limits[1], image_scale)
    save_field_image(errors[0], folder / f"05_F-OAE_error_MSE_{values['foae']:.3e}.png", error_limits[0], error_limits[1], image_scale)
    save_field_image(errors[1], folder / f"06_OAE_error_MSE_{values['oae']:.3e}.png", error_limits[0], error_limits[1], image_scale)
    save_field_image(errors[2], folder / f"07_AE_error_MSE_{values['ae']:.3e}.png", error_limits[0], error_limits[1], image_scale)
    save_colorbar(folder / "field_colorbar.png", *field_limits)
    save_colorbar(folder / "error_colorbar.png", *error_limits)
    return {
        "mse_foae": values["foae"],
        "mse_oae": values["oae"],
        "mse_ae": values["ae"],
        "field_vmin": field_limits[0],
        "field_vmax": field_limits[1],
        "error_vmin": error_limits[0],
        "error_vmax": error_limits[1],
    }


def load_pickled_model(path: Path, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(path)
    model = torch.load(path, map_location=device, weights_only=False)
    if hasattr(model, "eval"):
        model = model.to(device)
        model.eval()
    return model


def load_ae_reduction_model(ae_root: Path, device: torch.device, model_name: str):
    reduction_dir = ae_root / "reduction"
    sys.path.insert(0, str(reduction_dir))
    try:
        __import__("nn")
        model_path = resolve_existing(reduction_dir / "models", model_name)
        model = load_pickled_model(model_path, device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, model_path
    finally:
        if str(reduction_dir) in sys.path:
            sys.path.remove(str(reduction_dir))


def load_trunk(handle: h5py.File, key: str, device: torch.device) -> torch.Tensor:
    if key not in handle:
        raise KeyError(f"Required trunk dataset not found: {key}")
    return torch.from_numpy(np.asarray(handle[key][:], dtype=np.float32)).to(device)


def decode_dpn_batch(z: torch.Tensor, trunk: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
    if trunk.ndim == 3:
        if trunk.shape[0] == z.shape[-1]:
            field = torch.einsum("bi,ihw->bhw", z, trunk)
        elif trunk.shape[-1] == z.shape[-1]:
            field = torch.einsum("bi,hwi->bhw", z, trunk)
        else:
            raise ValueError(f"Cannot match latent dimension {z.shape[-1]} with trunk shape {tuple(trunk.shape)}.")
        return field
    if trunk.ndim == 2:
        if trunk.shape[0] == z.shape[-1]:
            flat = torch.einsum("bi,io->bo", z, trunk)
        elif trunk.shape[1] == z.shape[-1]:
            flat = torch.einsum("bi,oi->bo", z, trunk)
        else:
            raise ValueError(f"Cannot match latent dimension {z.shape[-1]} with trunk shape {tuple(trunk.shape)}.")
        return flat.reshape(z.shape[0], spatial_shape[0], spatial_shape[1])
    raise ValueError(f"Unsupported trunk shape {tuple(trunk.shape)}.")


def decode_ae_batch(z: torch.Tensor, model, spatial_shape: tuple[int, int]) -> torch.Tensor:
    out = model.decode(z)
    if out.ndim == 4 and out.shape[1] == 1:
        return out[:, 0]
    if out.ndim == 3:
        return out
    if out.ndim == 2:
        return out.reshape(z.shape[0], spatial_shape[0], spatial_shape[1])
    raise ValueError(f"Unsupported AE decode output shape {tuple(out.shape)}.")


def make_temporal_input(initial_latent: torch.Tensor, n_times: int) -> torch.Tensor:
    return initial_latent.reshape(initial_latent.shape[0], 1, initial_latent.shape[-1], 1).repeat(1, n_times, 1, 1)


@torch.no_grad()
def predict_latent_trajectory(model, initial_latent: np.ndarray, n_times: int, device: torch.device) -> torch.Tensor:
    latent = torch.from_numpy(np.asarray(initial_latent, dtype=np.float32)).to(device)
    if latent.ndim == 1:
        latent = latent[None]
    x = make_temporal_input(latent, n_times).permute(0, 2, 1, 3)
    out = model(x).permute(0, 2, 1)
    if out.shape[1] != n_times:
        raise ValueError(f"Temporal model returned {tuple(out.shape)}, expected time dimension {n_times}.")
    return out


def field_from_dataset(dataset: h5py.Dataset, case: int, time_idx: int) -> np.ndarray:
    value = np.asarray(dataset[case, time_idx], dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"Expected full field dataset slice [H,W], got {value.shape}.")
    return value


def sample_pairs(rng: np.random.Generator, start: int, stop: int, n_samples: int, n_times: int) -> list[tuple[int, int]]:
    cases = rng.integers(start, stop, size=n_samples)
    times = rng.integers(0, n_times, size=n_samples)
    return sorted((int(case), int(time_idx)) for case, time_idx in zip(cases, times))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    dm_root = autodetect_root("DM-2dns", args.dm_root)
    ae_root = autodetect_root("AE-2dns", args.ae_root)
    dm_data_path = dm_root / "reduction" / "data" / "data.hdf5"
    ae_data_path = ae_root / "reduction" / "data" / "data.hdf5"
    if not dm_data_path.exists():
        raise FileNotFoundError(dm_data_path)
    if not ae_data_path.exists():
        raise FileNotFoundError(ae_data_path)

    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    temporal_paths = {
        "foae": resolve_existing(dm_root / "latent_pred" / "model", FOAE_TEMPORAL_MODEL),
        "oae": resolve_existing(dm_root / "latent_pred" / "model", OAE_TEMPORAL_MODEL),
        "ae": resolve_existing(ae_root / "latent_pred" / "model", AE_TEMPORAL_MODEL),
    }
    temporal = {method: load_pickled_model(path, device) for method, path in temporal_paths.items()}
    ae_decoder, ae_reduction_path = load_ae_reduction_model(ae_root, device, AE_REDUCTION_MODEL)

    with h5py.File(dm_data_path, "r") as dm, h5py.File(ae_data_path, "r") as ae_data:
        full = dm[FULL_KEY]
        if full.ndim != 4:
            raise ValueError(f"{FULL_KEY} must have shape [N,T,H,W], got {full.shape}.")
        n_cases, n_times, height, width = full.shape
        heldout_start = max(0, args.heldout_start)
        heldout_stop = min(n_cases, args.heldout_stop)
        if heldout_stop <= heldout_start:
            raise ValueError(f"Invalid heldout range [{heldout_start}, {heldout_stop}) for {n_cases} cases.")
        spatial_shape = (height, width)

        latents = {
            "foae": np.asarray(dm[FOAE_LATENT_KEY][:], dtype=np.float32),
            "oae": np.asarray(dm[OAE_LATENT_KEY][:], dtype=np.float32),
            "ae": np.asarray(ae_data[AE_LATENT_KEY][:], dtype=np.float32),
        }
        trunks = {
            "foae": load_trunk(dm, FOAE_TRUNK_KEY, device),
            "oae": load_trunk(dm, OAE_TRUNK_KEY, device),
        }

        manifest: dict[str, object] = {
            "dm_root": str(dm_root),
            "ae_root": str(ae_root),
            "dm_data": str(dm_data_path),
            "ae_data": str(ae_data_path),
            "full_key": FULL_KEY,
            "full_shape": list(full.shape),
            "latent_keys": {
                "foae": FOAE_LATENT_KEY,
                "oae": OAE_LATENT_KEY,
                "ae": AE_LATENT_KEY,
            },
            "trunk_keys": {
                "foae": FOAE_TRUNK_KEY,
                "oae": OAE_TRUNK_KEY,
            },
            "reduction_models": {
                "foae": str(resolve_existing(dm_root / "reduction" / "models", FOAE_REDUCTION_MODEL)),
                "oae": str(resolve_existing(dm_root / "reduction" / "models", OAE_REDUCTION_MODEL)),
                "ae": str(ae_reduction_path),
            },
            "temporal_models": {method: str(path) for method, path in temporal_paths.items()},
            "image_size_pixels": [width * args.image_scale, height * args.image_scale],
            "field_shape": [height, width],
            "cmap": CMAP_NAME,
            "colorbar": "two horizontal scales per folder: one shared by response fields and one shared by absolute-error maps",
            "heldout_cases_zero_based": [heldout_start, heldout_stop - 1],
            "reconstruction": [],
            "latent_prediction": [],
        }

        for case, time_idx in sample_pairs(rng, heldout_start, heldout_stop, args.n_samples, n_times):
            benchmark = field_from_dataset(full, case, time_idx)
            z_foae = torch.from_numpy(latents["foae"][case, time_idx : time_idx + 1]).to(device)
            z_oae = torch.from_numpy(latents["oae"][case, time_idx : time_idx + 1]).to(device)
            z_ae = torch.from_numpy(latents["ae"][case, time_idx : time_idx + 1]).to(device)
            foae = decode_dpn_batch(z_foae, trunks["foae"], spatial_shape)[0].detach().cpu().numpy()
            oae = decode_dpn_batch(z_oae, trunks["oae"], spatial_shape)[0].detach().cpu().numpy()
            ae = decode_ae_batch(z_ae, ae_decoder, spatial_shape)[0].detach().cpu().numpy()
            folder = args.output_dir / "reconstruction" / f"case_{case:04d}_t_{time_idx:03d}"
            row = save_sample_outputs(folder, benchmark, foae, oae, ae, args.image_scale)
            row.update({"case": case, "time": time_idx})
            manifest["reconstruction"].append(row)
            print(json.dumps({"stage": "reconstruction", "case": case, "time": time_idx}), flush=True)

        latent_case = args.latent_case
        if latent_case is None:
            latent_case = int(rng.integers(heldout_start, heldout_stop))
        times = sorted(int(value) for value in rng.choice(np.arange(n_times), size=args.n_samples, replace=False))
        predicted_latents = {
            method: predict_latent_trajectory(temporal[method], latents[method][latent_case, 0], n_times, device)
            for method in ("foae", "oae", "ae")
        }
        for time_idx in times:
            benchmark = field_from_dataset(full, latent_case, time_idx)
            foae = decode_dpn_batch(predicted_latents["foae"][:, time_idx], trunks["foae"], spatial_shape)[0].detach().cpu().numpy()
            oae = decode_dpn_batch(predicted_latents["oae"][:, time_idx], trunks["oae"], spatial_shape)[0].detach().cpu().numpy()
            ae = decode_ae_batch(predicted_latents["ae"][:, time_idx], ae_decoder, spatial_shape)[0].detach().cpu().numpy()
            folder = args.output_dir / "latent_prediction" / f"case_{latent_case:04d}_t_{time_idx:03d}"
            row = save_sample_outputs(folder, benchmark, foae, oae, ae, args.image_scale)
            row.update({"case": latent_case, "time": time_idx})
            manifest["latent_prediction"].append(row)
            print(json.dumps({"stage": "latent_prediction", "case": latent_case, "time": time_idx}), flush=True)

    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"output": str(args.output_dir), "latent_case": latent_case}), flush=True)


if __name__ == "__main__":
    main()
