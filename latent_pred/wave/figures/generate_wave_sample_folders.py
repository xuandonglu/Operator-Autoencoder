from __future__ import annotations

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
OAE_ROOT = THIS_FILE.parents[3]
WORKSPACE = OAE_ROOT.parent
OUT_DIR = THIS_FILE.parent
DM_ROOT = OAE_ROOT
AE_ROOT = OAE_ROOT
DM_DATA = OAE_ROOT / "reduction" / "data" / "data_wave.h5df"
AE_DATA = DM_DATA
DM_LATENT_DIR = DM_ROOT / "latent_pred" / "wave"
AE_LATENT_DIR = AE_ROOT / "latent_pred" / "wave"
AE_REDUCTION_DIR = AE_ROOT / "reduction"
AE_REDUCTION_MODEL = OAE_ROOT / "reduction" / "models" / "AE_d30_wave.pth"
F_OAE_TEMPORAL = OAE_ROOT / "latent_pred" / "wave" / "models" / "FOAE_d30_wave_latent.pt"
OAE_TEMPORAL = OAE_ROOT / "latent_pred" / "wave" / "models" / "OAE_d30_wave_latent.pt"
AE_TEMPORAL = OAE_ROOT / "latent_pred" / "wave" / "models" / "AE_d30_wave_latent.pt"

FULL_KEY = "wave_5000_501_256"
F_OAE_LATENT_KEY = "latent_DPN_d30_fno"
F_OAE_TRUNK_KEY = "trunk_DPN_d30_fno"
OAE_LATENT_KEY = "latent_DPN_d30"
OAE_TRUNK_KEY = "trunk_DPN_d30"
AE_LATENT_KEY = "latent_AE_d30"
SELECTED_SAMPLES = [4902, 4905, 4936, 4970, 4993]

CMAP_NAME = "RdBu"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512


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

    def compl_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
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
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


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
        x = self.fc0(torch.cat((x, grid), dim=-1)).permute(0, 3, 1, 2)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))
        return self.fc2(F.gelu(self.fc1(x.permute(0, 2, 3, 1)))).squeeze(-1)

    @staticmethod
    def get_grid(shape: torch.Size, device: torch.device) -> torch.Tensor:
        batch_size, size_x, size_y = shape[0], shape[1], shape[2]
        grid_x = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1).repeat(batch_size, 1, size_y, 1)
        grid_y = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1).repeat(batch_size, size_x, 1, 1)
        return torch.cat((grid_x, grid_y), dim=-1)


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


def save_field_image(array: np.ndarray, path: Path, vmin: float, vmax: float) -> None:
    rgba = plt.get_cmap(CMAP_NAME)(colors.Normalize(vmin=vmin, vmax=vmax)(array), bytes=True)
    Image.fromarray(rgba).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BICUBIC).save(path)


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
    cb.ax.tick_params(labelsize=12, length=0, pad=3)
    cb.outline.set_linewidth(0.8)
    if exponent != 0:
        fig.text(0.64, 0.62, rf"$\times 10^{{{exponent}}}$", fontsize=10, ha="left", va="center")
    fig.savefig(path, transparent=False)
    plt.close(fig)


def mse(prediction: np.ndarray, benchmark: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - benchmark)))


def save_sample_outputs(
    sample_dir: Path,
    benchmark: np.ndarray,
    foae: np.ndarray,
    oae: np.ndarray,
    ae: np.ndarray,
) -> dict[str, float]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    for old in sample_dir.glob("*.png"):
        old.unlink()
    errors = [np.abs(foae - benchmark), np.abs(oae - benchmark), np.abs(ae - benchmark)]
    fields = [benchmark, foae, oae, ae]
    field_limits = rounded_limits(float(min(field.min() for field in fields)), float(max(field.max() for field in fields)))
    error_limits = rounded_limits(0.0, float(max(error.max() for error in errors)), force_zero_bottom=True)
    values = {"foae": mse(foae, benchmark), "oae": mse(oae, benchmark), "ae": mse(ae, benchmark)}

    save_field_image(benchmark, sample_dir / "01_benchmark.png", field_limits[0], field_limits[1])
    save_field_image(foae, sample_dir / "02_F-OAE.png", field_limits[0], field_limits[1])
    save_field_image(oae, sample_dir / "03_OAE.png", field_limits[0], field_limits[1])
    save_field_image(ae, sample_dir / "04_AE.png", field_limits[0], field_limits[1])
    save_field_image(errors[0], sample_dir / f"05_F-OAE_error_MSE_{values['foae']:.3e}.png", error_limits[0], error_limits[1])
    save_field_image(errors[1], sample_dir / f"06_OAE_error_MSE_{values['oae']:.3e}.png", error_limits[0], error_limits[1])
    save_field_image(errors[2], sample_dir / f"07_AE_error_MSE_{values['ae']:.3e}.png", error_limits[0], error_limits[1])
    save_colorbar(sample_dir / "field_colorbar.png", *field_limits)
    save_colorbar(sample_dir / "error_colorbar.png", *error_limits)
    return {
        "mse_foae": values["foae"],
        "mse_oae": values["oae"],
        "mse_ae": values["ae"],
        "field_vmin": field_limits[0],
        "field_vmax": field_limits[1],
        "error_vmin": error_limits[0],
        "error_vmax": error_limits[1],
    }


def load_ae_reduction_model(device: torch.device) -> nn.Module:
    sys.path.insert(0, str(AE_REDUCTION_DIR))
    from checkpoint_utils import load_reduction_checkpoint

    model = load_reduction_checkpoint(AE_REDUCTION_MODEL, 'wave', 'AE', device)
    model.eval()
    return model


def load_temporal_model(path: Path, device: torch.device) -> nn.Module:
    model = torch.load(path, map_location=device, weights_only=False).to(device)
    model.eval()
    return model


def make_input(initial: torch.Tensor, n_times: int) -> torch.Tensor:
    return initial.reshape(initial.shape[0], 1, initial.shape[-1], 1).repeat(1, n_times, 1, 1)


def predict_latent(model: nn.Module, initial: torch.Tensor, n_times: int) -> torch.Tensor:
    return model(make_input(initial, n_times).permute(0, 2, 1, 3)).permute(0, 2, 1)


def decode_dpn(latent: torch.Tensor, trunk: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bti,io->bto", latent, trunk)


def decode_ae(latent: torch.Tensor, ae: nn.Module) -> torch.Tensor:
    return ae.decode(latent.reshape(-1, latent.shape[-1])).reshape(latent.shape[0], latent.shape[1], -1)


@torch.no_grad()
def evaluate_all(
    device: torch.device,
    ae: nn.Module,
    temporal: dict[str, nn.Module],
    latents: dict[str, np.ndarray],
    trunks: dict[str, torch.Tensor],
    benchmark: h5py.Dataset,
    batch_size: int = 10,
) -> dict[str, dict[str, float]]:
    totals = {
        stage: {method: {"sse": 0.0, "count": 0} for method in ("foae", "oae", "ae")}
        for stage in ("reconstruction", "latent_prediction")
    }
    for case_start in range(4900, 5000, batch_size):
        case_end = min(case_start + batch_size, 5000)
        truth = torch.from_numpy(np.asarray(benchmark[case_start:case_end], dtype=np.float32)).to(device)
        true_latent = {
            method: torch.from_numpy(latents[method][case_start:case_end]).to(device)
            for method in latents
        }
        reconstructed = {
            "foae": decode_dpn(true_latent["foae"], trunks["foae"]),
            "oae": decode_dpn(true_latent["oae"], trunks["oae"]),
            "ae": decode_ae(true_latent["ae"], ae),
        }
        predicted_latent = {
            method: predict_latent(temporal[method], true_latent[method][:, 0], truth.shape[1])
            for method in latents
        }
        predicted = {
            "foae": decode_dpn(predicted_latent["foae"], trunks["foae"]),
            "oae": decode_dpn(predicted_latent["oae"], trunks["oae"]),
            "ae": decode_ae(predicted_latent["ae"], ae),
        }
        for stage, outputs in (("reconstruction", reconstructed), ("latent_prediction", predicted)):
            for method, output in outputs.items():
                totals[stage][method]["sse"] += float(torch.square(output - truth).sum().cpu())
                totals[stage][method]["count"] += truth.numel()
    return {
        stage: {method: values["sse"] / values["count"] for method, values in methods.items()}
        for stage, methods in totals.items()
    }


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ae = load_ae_reduction_model(device)
    temporal = {
        "foae": load_temporal_model(F_OAE_TEMPORAL, device),
        "oae": load_temporal_model(OAE_TEMPORAL, device),
        "ae": load_temporal_model(AE_TEMPORAL, device),
    }
    with h5py.File(DM_DATA, "r") as dm, h5py.File(AE_DATA, "r") as ae_data:
        benchmark = dm[FULL_KEY]
        latents = {
            "foae": np.asarray(dm[F_OAE_LATENT_KEY][:], dtype=np.float32),
            "oae": np.asarray(dm[OAE_LATENT_KEY][:], dtype=np.float32),
            "ae": np.asarray(ae_data[AE_LATENT_KEY][:], dtype=np.float32),
        }
        trunks = {
            "foae": torch.from_numpy(np.asarray(dm[F_OAE_TRUNK_KEY][:], dtype=np.float32)).to(device),
            "oae": torch.from_numpy(np.asarray(dm[OAE_TRUNK_KEY][:], dtype=np.float32)).to(device),
        }
        summary = {
            "selected_sample_indices_zero_based": SELECTED_SAMPLES,
            "image_size_pixels": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "field_shape": [501, 256],
            "cmap": CMAP_NAME,
            "colorbar": "two horizontal scales per folder: one for response fields and one for absolute-error maps",
            "validation_cases_zero_based": [4900, 4999],
            "validation_overall_mse": evaluate_all(device, ae, temporal, latents, trunks, benchmark),
            "samples": {},
        }
        for index in SELECTED_SAMPLES:
            truth = np.asarray(benchmark[index], dtype=np.float32)
            true_latent = {method: torch.from_numpy(latents[method][index : index + 1]).to(device) for method in latents}
            reconstructed = {
                "foae": decode_dpn(true_latent["foae"], trunks["foae"])[0].cpu().numpy(),
                "oae": decode_dpn(true_latent["oae"], trunks["oae"])[0].cpu().numpy(),
                "ae": decode_ae(true_latent["ae"], ae)[0].cpu().numpy(),
            }
            predicted_latent = {
                method: predict_latent(temporal[method], true_latent[method][:, 0], truth.shape[0])
                for method in latents
            }
            predicted = {
                "foae": decode_dpn(predicted_latent["foae"], trunks["foae"])[0].cpu().numpy(),
                "oae": decode_dpn(predicted_latent["oae"], trunks["oae"])[0].cpu().numpy(),
                "ae": decode_ae(predicted_latent["ae"], ae)[0].cpu().numpy(),
            }
            root = OUT_DIR / f"sample_{index:04d}"
            summary["samples"][str(index)] = {
                "reconstruction": save_sample_outputs(root / "reconstruction", truth, reconstructed["foae"], reconstructed["oae"], reconstructed["ae"]),
                "latent_prediction": save_sample_outputs(root / "latent_prediction", truth, predicted["foae"], predicted["oae"], predicted["ae"]),
            }
            print({"sample": index, "done": True}, flush=True)

    (OUT_DIR / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT_DIR), "samples": SELECTED_SAMPLES}), flush=True)


if __name__ == "__main__":
    main()
