from __future__ import annotations

import importlib.util
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
import torch.nn.functional as F
from torch.utils.data import DataLoader


THIS_FILE = Path(__file__).resolve()
RUN_DIR = THIS_FILE.parent
OAE_ROOT = THIS_FILE.parents[2]
REDUCTION_DIR = OAE_ROOT / "reduction"
MODELS_DIR = REDUCTION_DIR / "models"
DATA_PATH = REDUCTION_DIR / "data" / "data_cylinder.h5df"
TRAIN_SCRIPT = RUN_DIR / "train_latent_hist10_to10_2dfno.py"
OUTPUT_DIR = RUN_DIR / "sample_folders"

FULL_KEY = "uxuy_50_300_1_768_512"
HISTORY = 10
FUTURE_STEPS = 10
LATENT_SPLIT = (80, 40, 20)
CMAP_NAME = "RdBu"  # Low values red, high values blue, white near the middle.
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 512
SEED = 20260531

METHODS = {
    "foae": {
        "latent": "latent_DPN_d80_fno_cylinder",
        "trunk": "trunk_DPN_d80_fno_cylinder",
        "checkpoints": [
            RUN_DIR / "models" / "FOAE_d140_cylinder_latent.pt",
        ],
    },
    "oae": {
        "latent": "latent_DPN_d80_cylinder",
        "trunk": "trunk_DPN_d80_cylinder",
        "checkpoints": [
            RUN_DIR / "models" / "OAE_d140_cylinder_latent.pt",
        ],
    },
    "ae": {
        "latent": "latent_AE_d140_cylinder",
        "checkpoints": [
            RUN_DIR / "models" / "AE_d140_cylinder_latent.pt",
        ],
    },
}


def import_train_module():
    spec = importlib.util.spec_from_file_location("cylinder_hist10_to10_train", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = import_train_module()


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    rgba = plt.get_cmap(CMAP_NAME)(colors.Normalize(vmin=vmin, vmax=vmax)(array.T), bytes=True)
    image = Image.fromarray(rgba)
    if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        raise ValueError(f"Unexpected cylinder image size {image.size}; expected {(IMAGE_WIDTH, IMAGE_HEIGHT)}.")
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


def as_field2d(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    array = np.asarray(array, dtype=np.float32).squeeze()
    if array.shape != (IMAGE_WIDTH, IMAGE_HEIGHT):
        raise ValueError(f"Unexpected cylinder field shape {array.shape}; expected {(IMAGE_WIDTH, IMAGE_HEIGHT)}.")
    return array


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
) -> dict[str, float]:
    clean_pngs(folder)
    fields = [benchmark, foae, oae, ae]
    errors = [np.abs(foae - benchmark), np.abs(oae - benchmark), np.abs(ae - benchmark)]
    field_limits = rounded_limits(float(min(field.min() for field in fields)), float(max(field.max() for field in fields)))
    error_limits = rounded_limits(0.0, float(max(error.max() for error in errors)), force_zero_bottom=True)
    values = {"foae": mse(foae, benchmark), "oae": mse(oae, benchmark), "ae": mse(ae, benchmark)}

    save_field_image(benchmark, folder / "01_benchmark.png", field_limits[0], field_limits[1])
    save_field_image(foae, folder / "02_F-OAE.png", field_limits[0], field_limits[1])
    save_field_image(oae, folder / "03_OAE.png", field_limits[0], field_limits[1])
    save_field_image(ae, folder / "04_AE.png", field_limits[0], field_limits[1])
    save_field_image(errors[0], folder / f"05_F-OAE_error_MSE_{values['foae']:.3e}.png", error_limits[0], error_limits[1])
    save_field_image(errors[1], folder / f"06_OAE_error_MSE_{values['oae']:.3e}.png", error_limits[0], error_limits[1])
    save_field_image(errors[2], folder / f"07_AE_error_MSE_{values['ae']:.3e}.png", error_limits[0], error_limits[1])
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


def load_temporal_model(method: str, device: torch.device):
    candidates = []
    for checkpoint_path in METHODS[method]["checkpoints"]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        candidates.append((float(checkpoint["metrics"]["test_mse_norm"]), Path(checkpoint_path), checkpoint))
    heldout_mse_norm, checkpoint_path, checkpoint = min(candidates, key=lambda item: item[0])
    config = checkpoint["model_config"]
    model = TRAIN.LatentHistoryToFutureFNO2d(
        latent_size=int(config["latent_size"]),
        history=int(config["history"]),
        future_steps=int(config["future_steps"]),
        width=int(config["width"]),
        layers=int(config["layers"]),
        modes_latent=int(config["modes_latent"]),
        modes_time=int(config["modes_time"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {
        "model": model,
        "mean": np.asarray(checkpoint["mean"], dtype=np.float32),
        "std": np.asarray(checkpoint["std"], dtype=np.float32),
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "checkpoint_heldout_mse_norm": heldout_mse_norm,
    }


@torch.no_grad()
def predict_latent(predictor: dict[str, object], z_window: np.ndarray, device: torch.device) -> torch.Tensor:
    mean = torch.from_numpy(predictor["mean"]).reshape(1, 1, -1).to(device)
    std = torch.from_numpy(predictor["std"]).reshape(1, 1, -1).to(device)
    array = np.asarray(z_window, dtype=np.float32)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"Expected latent windows shaped [batch, history, latent], got {array.shape}.")
    x = torch.from_numpy(array).to(device)
    return predictor["model"]((x - mean) / std) * std + mean


def load_ae_decoder(device: torch.device):
    sys.path.insert(0, str(REDUCTION_DIR))
    import nn_ae_cylinder  # noqa: F401

    model = torch.load(MODELS_DIR / "AE_d140_cylinder.pth", map_location=device, weights_only=False).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_trunks(handle: h5py.File, prefix: str, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        level: torch.from_numpy(np.asarray(handle[f"{prefix}_{level}"][:], dtype=np.float32)).to(device)
        for level in ("f", "m", "c")
    }


@torch.no_grad()
def decode_dpn_batch(z: torch.Tensor, trunks: dict[str, torch.Tensor]) -> torch.Tensor:
    zf, zm, zc = torch.split(z, LATENT_SPLIT, dim=-1)
    yf = torch.einsum("bi,ixy->bxy", zf, trunks["f"])
    ym = torch.einsum("bi,ixy->bxy", zm, trunks["m"])
    yc = torch.einsum("bi,ixy->bxy", zc, trunks["c"])
    ym = F.interpolate(ym[:, None], size=yf.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    yc = F.interpolate(yc[:, None], size=yf.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return (yf + ym + yc)[:, None]


@torch.no_grad()
def decode_dpn(z: torch.Tensor, trunks: dict[str, torch.Tensor]) -> np.ndarray:
    return as_field2d(decode_dpn_batch(z, trunks))


@torch.no_grad()
def decode_ae(z: torch.Tensor, model) -> np.ndarray:
    return as_field2d(model.decode(z))


@torch.no_grad()
def evaluate_last_output_field_mse(
    full: h5py.Dataset,
    predictors: dict[str, dict[str, object]],
    latents: dict[str, np.ndarray],
    trunks: dict[str, dict[str, torch.Tensor]],
    ae_decoder,
    device: torch.device,
) -> dict[str, float]:
    stop = full.shape[1] - HISTORY - FUTURE_STEPS + 1
    starts = np.arange(stop)
    targets = starts + HISTORY + FUTURE_STEPS - 1
    values = {}
    for method, predictor in predictors.items():
        batch_size = 4 if method == "ae" else 16
        sse = 0.0
        count = 0
        for case in range(40, 50):
            for offset in range(0, stop, batch_size):
                batch_starts = starts[offset : offset + batch_size]
                batch_targets = targets[offset : offset + batch_size]
                windows = np.stack(
                    [latents[method][case, start : start + HISTORY] for start in batch_starts],
                    axis=0,
                )
                predicted_z = predict_latent(predictor, windows, device)[:, -1]
                predicted = ae_decoder.decode(predicted_z) if method == "ae" else decode_dpn_batch(predicted_z, trunks[method])
                truth = torch.from_numpy(np.asarray(full[case, batch_targets, 0], dtype=np.float32))[:, None].to(device)
                sse += float(torch.square(predicted - truth).sum().cpu())
                count += truth.numel()
        values[method] = sse / max(1, count)
        print(json.dumps({"method": method, "last_output_field_mse": values[method]}), flush=True)
    return values


def evaluate_latent_metrics(
    predictors: dict[str, dict[str, object]],
    latents: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, dict[str, object]]:
    metrics = {}
    for method, predictor in predictors.items():
        values = TRAIN.evaluate(
            predictor["model"],
            latents[method],
            list(range(40, 50)),
            HISTORY,
            FUTURE_STEPS,
            predictor["mean"],
            predictor["std"],
            device,
            batch_size=256,
        )
        dataset = TRAIN.HistoryToFutureDataset(
            latents[method],
            list(range(40, 50)),
            HISTORY,
            FUTURE_STEPS,
            predictor["mean"],
            predictor["std"],
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        step_sse = torch.zeros(FUTURE_STEPS, dtype=torch.float64)
        step_count = 0
        for x, y in loader:
            pred = predictor["model"](x.to(device, non_blocking=True))
            diff = pred - y.to(device, non_blocking=True)
            step_sse += torch.square(diff).sum(dim=(0, 2)).detach().cpu().double()
            step_count += diff.shape[0] * diff.shape[2]
        values["per_step_mse_norm"] = (step_sse / max(1, step_count)).numpy().astype(float).tolist()
        metrics[method] = values
    return metrics


def plot_normalized_latent_metrics(metrics: dict[str, dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=180)
    x = np.arange(1, FUTURE_STEPS + 1)
    for method, label in (("foae", "F-OAE"), ("oae", "OAE"), ("ae", "AE")):
        values = np.asarray(metrics[method]["per_step_mse_norm"], dtype=np.float64)
        ax.plot(x, values, marker="o", linewidth=1.8, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Predicted step within each truth-input window")
    ax.set_ylabel("Normalized latent MSE")
    ax.set_xticks(x)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "latent_mse_by_future_step.png")
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    device = resolve_device()
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictors = {method: load_temporal_model(method, device) for method in METHODS}
    ae_decoder = load_ae_decoder(device)

    with h5py.File(DATA_PATH, "r") as handle:
        full = handle[FULL_KEY]
        latents = {
            method: np.asarray(handle[info["latent"]][:], dtype=np.float32)
            for method, info in METHODS.items()
        }
        trunks = {
            method: load_trunks(handle, str(info["trunk"]), device)
            for method, info in METHODS.items()
            if "trunk" in info
        }
        latent_metrics = evaluate_latent_metrics(predictors, latents, device)
        last_output_field_mse = evaluate_last_output_field_mse(
            full,
            predictors,
            latents,
            trunks,
            ae_decoder,
            device,
        )
        manifest: dict[str, object] = {
            "mapping": "truth z[t:t+10] -> z[t+10:t+20]; no autoregressive feedback",
            "displayed_prediction": "the farthest output in each 10-step prediction window, z[t+19]",
            "train_cases_zero_based": [0, 39],
            "heldout_cases_zero_based": [40, 49],
            "image_size_pixels": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "field_shape": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "cmap": CMAP_NAME,
            "colorbar": "two horizontal scales per folder: one shared by response fields and one shared by absolute-error maps",
            "checkpoints": {
                method: {
                    "path": predictor["checkpoint"],
                    "epoch": predictor["epoch"],
                    "heldout_mse_norm": predictor["checkpoint_heldout_mse_norm"],
                }
                for method, predictor in predictors.items()
            },
            "heldout_latent_metrics": latent_metrics,
            "heldout_last_output_field_mse": last_output_field_mse,
            "reconstruction": [],
            "latent_prediction": [],
        }

        reconstruction_pairs = sorted(
            [(int(case), int(time)) for case, time in zip(rng.integers(40, 50, 5), rng.integers(0, 300, 5))]
        )
        for case, time_idx in reconstruction_pairs:
            benchmark = as_field2d(np.asarray(full[case, time_idx, 0], dtype=np.float32))
            foae = decode_dpn(torch.from_numpy(latents["foae"][case, time_idx])[None].to(device), trunks["foae"])
            oae = decode_dpn(torch.from_numpy(latents["oae"][case, time_idx])[None].to(device), trunks["oae"])
            ae = decode_ae(torch.from_numpy(latents["ae"][case, time_idx])[None].to(device), ae_decoder)
            folder = OUTPUT_DIR / "reconstruction" / f"case_{case:02d}_t_{time_idx:03d}"
            row = save_sample_outputs(folder, benchmark, foae, oae, ae)
            row.update({"case": case, "time": time_idx})
            manifest["reconstruction"].append(row)

        latent_case = int(rng.integers(40, 50))
        valid_starts = np.arange(full.shape[1] - HISTORY - FUTURE_STEPS + 1)
        input_starts = sorted(int(value) for value in rng.choice(valid_starts, size=5, replace=False))
        for input_start in input_starts:
            input_end = input_start + HISTORY - 1
            target_t = input_start + HISTORY + FUTURE_STEPS - 1
            benchmark = as_field2d(np.asarray(full[latent_case, target_t, 0], dtype=np.float32))
            predicted = {
                method: predict_latent(predictors[method], latents[method][latent_case, input_start : input_start + HISTORY], device)[
                    :, -1
                ]
                for method in METHODS
            }
            foae = decode_dpn(predicted["foae"], trunks["foae"])
            oae = decode_dpn(predicted["oae"], trunks["oae"])
            ae = decode_ae(predicted["ae"], ae_decoder)
            folder = (
                OUTPUT_DIR
                / "latent_prediction"
                / f"case_{latent_case:02d}_input_{input_start:03d}-{input_end:03d}_target_{target_t:03d}"
            )
            row = save_sample_outputs(folder, benchmark, foae, oae, ae)
            row.update({"case": latent_case, "input_start": input_start, "input_end": input_end, "target_t": target_t})
            manifest["latent_prediction"].append(row)

    plot_normalized_latent_metrics(latent_metrics)
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"output": str(OUTPUT_DIR), "latent_case": latent_case, "input_starts": input_starts}), flush=True)


if __name__ == "__main__":
    main()
