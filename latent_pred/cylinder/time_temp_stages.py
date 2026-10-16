from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
CYLINDER_DIR = THIS_FILE.parent
PROJECT_ROOT = CYLINDER_DIR.parents[1]
REDUCTION_DIR = PROJECT_ROOT / "reduction"
REDUCTION_MODELS = REDUCTION_DIR / "models"
DEFAULT_DATA_PATH = REDUCTION_DIR / "data" / "data_cylinder_temp_xmask.h5df"
DEFAULT_OUTPUT_DIR = CYLINDER_DIR / "figures" / "temp_timing"
TRAIN_SCRIPT = CYLINDER_DIR / "train_latent_hist10_to10_2dfno.py"
FULL_KEY = "uxuy_50_300_1_768_512"
HISTORY = 10
FUTURE_STEPS = 10

METHODS = {
    "AE": {
        "reduction_model": REDUCTION_MODELS / "AE_d140_cylinder.pth",
        "reduction_module": "nn_ae_cylinder",
        "latent_model": CYLINDER_DIR / "models" / "AE_d140_cylinder_latent.pt",
        "reconstruction": "ae",
    },
    "FOAE": {
        "reduction_model": REDUCTION_MODELS / "FOAE_d140_cylinder.pth",
        "reduction_module": "nn_cylinder",
        "latent_model": CYLINDER_DIR / "models" / "FOAE_d140_cylinder_latent.pt",
        "reconstruction": "multiscale_trunk",
        "trunk_prefix": "temp_trunk_FOAE_d140_cylinder",
    },
    "OAE": {
        "reduction_model": REDUCTION_MODELS / "OAE_d140_cylinder.pth",
        "reduction_module": "nn_cylinder",
        "latent_model": CYLINDER_DIR / "models" / "OAE_d140_cylinder_latent.pt",
        "reconstruction": "multiscale_trunk",
        "trunk_prefix": "temp_trunk_OAE_d140_cylinder",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Time temporary cylinder 10-to-10 inference stages. Encode and reconstruction "
            "process the same number of field slices: n_windows * 10."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=["AE", "FOAE"])
    parser.add_argument("--slice-batch-size", type=int, default=4)
    parser.add_argument("--window-batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn):
    sync(device)
    start = perf_counter()
    result = fn()
    sync(device)
    return perf_counter() - start, result


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def import_train_module():
    spec = importlib.util.spec_from_file_location("cylinder_latent_train_for_timing", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = import_train_module()


def patch_reduction_module(module_name: str) -> None:
    sys.path.insert(0, str(REDUCTION_DIR))
    module = __import__(module_name)
    sys.modules["nn"] = module
    for name in ("AE", "DPN", "FNO1", "FNO2", "ResidualBlock", "SpectralConv2d"):
        if hasattr(module, name):
            globals()[name] = getattr(module, name)


def load_reduction_model(method: str, device: torch.device):
    config = METHODS[method]
    patch_reduction_module(config["reduction_module"])
    model = torch_load(config["reduction_model"], device).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_latent_model(method: str, device: torch.device):
    checkpoint = torch_load(METHODS[method]["latent_model"], torch.device("cpu"))
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "model": model,
        "mean": torch.from_numpy(np.asarray(checkpoint["mean"], dtype=np.float32)).reshape(1, 1, -1).to(device),
        "std": torch.from_numpy(np.asarray(checkpoint["std"], dtype=np.float32)).reshape(1, 1, -1).to(device),
    }


@torch.no_grad()
def encode_slices(model, slices: torch.Tensor, batch_size: int) -> torch.Tensor:
    parts = []
    for start in range(0, slices.shape[0], batch_size):
        parts.append(model.encode(slices[start : start + batch_size]))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def predict_latent_windows(predictor: dict[str, object], windows: torch.Tensor, batch_size: int) -> torch.Tensor:
    parts = []
    model = predictor["model"]
    mean = predictor["mean"]
    std = predictor["std"]
    for start in range(0, windows.shape[0], batch_size):
        x = windows[start : start + batch_size]
        parts.append(model((x - mean) / std) * std + mean)
    return torch.cat(parts, dim=0)


@torch.no_grad()
def decode_ae_slices(model, latent_slices: torch.Tensor, batch_size: int) -> None:
    for start in range(0, latent_slices.shape[0], batch_size):
        _ = model.decode(latent_slices[start : start + batch_size])


@torch.no_grad()
def decode_multiscale_slices(model, latent_slices: torch.Tensor, trunks: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_size: int) -> None:
    trunk_f, trunk_m, trunk_c = trunks
    for start in range(0, latent_slices.shape[0], batch_size):
        z = latent_slices[start : start + batch_size]
        zf, zm, zc = torch.split(z, [model.latent_dim_f, model.latent_dim_m, model.latent_dim_c], dim=-1)
        yf = torch.einsum("bi,ixy->bxy", zf, trunk_f)
        ym = torch.einsum("bi,ixy->bxy", zm, trunk_m)
        yc = torch.einsum("bi,ixy->bxy", zc, trunk_c)
        ym = torch.nn.functional.interpolate(ym[:, None], size=yf.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        yc = torch.nn.functional.interpolate(yc[:, None], size=yf.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        _ = (yf + ym + yc)[:, None]


def load_or_create_trunks(model, data_path: Path, prefix: str, device: torch.device):
    sys.path.insert(0, str(REDUCTION_MODELS))
    from load_test_utils import load_or_create_multiscale_trunk  # noqa: WPS433

    return load_or_create_multiscale_trunk(model, data_path, prefix, device)


def build_windows(fields: torch.Tensor) -> torch.Tensor:
    stop = fields.shape[0] - HISTORY - FUTURE_STEPS + 1
    if stop <= 0:
        raise ValueError(f"Need at least {HISTORY + FUTURE_STEPS} frames, got {fields.shape[0]}.")
    return torch.stack([fields[start : start + HISTORY] for start in range(stop)], dim=0)


def summarize(values: list[float]) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "runs_s": [float(value) for value in values],
    }


def write_csv(path: Path, summary: dict[str, object]) -> None:
    rows = []
    for method, result in summary["results"].items():
        for stage in ("encode", "latent_model", "reconstruction", "total"):
            rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "mean_s": result[stage]["mean_s"],
                    "std_s": result[stage]["std_s"],
                    "n_windows": summary["n_windows"],
                    "slices_per_encode": summary["slices_per_encode"],
                    "slices_per_reconstruction": summary["slices_per_reconstruction"],
                    "device": summary["device"],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    with h5py.File(args.data_path, "r") as handle:
        full = np.asarray(handle[FULL_KEY][0, :, 0], dtype=np.float32)
    fields = torch.from_numpy(full[:, None]).to(device)
    input_windows = build_windows(fields)
    n_windows = input_windows.shape[0]
    input_slices = input_windows.reshape(n_windows * HISTORY, *fields.shape[1:]).contiguous()

    results = {}
    for method in args.methods:
        reduction_model = load_reduction_model(method, device)
        latent_predictor = load_latent_model(method, device)
        trunks = None
        if METHODS[method]["reconstruction"] == "multiscale_trunk":
            trunks = load_or_create_trunks(reduction_model, args.data_path, METHODS[method]["trunk_prefix"], device)

        encode_times = []
        latent_times = []
        reconstruction_times = []
        total_times = []
        for repeat in range(args.warmup + args.repeats):
            encode_time, input_latent = timed(
                device,
                lambda model=reduction_model: encode_slices(model, input_slices, args.slice_batch_size),
            )
            input_latent_windows = input_latent.reshape(n_windows, HISTORY, -1).contiguous()
            latent_time, predicted_latent = timed(
                device,
                lambda model=latent_predictor, z=input_latent_windows: predict_latent_windows(model, z, args.window_batch_size),
            )
            output_latent_slices = predicted_latent.reshape(n_windows * FUTURE_STEPS, -1).contiguous()
            if METHODS[method]["reconstruction"] == "ae":
                reconstruction_time, _ = timed(
                    device,
                    lambda model=reduction_model, z=output_latent_slices: decode_ae_slices(model, z, args.slice_batch_size),
                )
            else:
                reconstruction_time, _ = timed(
                    device,
                    lambda model=reduction_model, z=output_latent_slices, basis=trunks: decode_multiscale_slices(model, z, basis, args.slice_batch_size),
                )

            total_time = encode_time + latent_time + reconstruction_time
            del input_latent, input_latent_windows, predicted_latent, output_latent_slices
            clear_cuda()
            if repeat >= args.warmup:
                encode_times.append(encode_time)
                latent_times.append(latent_time)
                reconstruction_times.append(reconstruction_time)
                total_times.append(total_time)
                print(
                    json.dumps(
                        {
                            "method": method,
                            "repeat": repeat - args.warmup,
                            "encode_s": encode_time,
                            "latent_model_s": latent_time,
                            "reconstruction_s": reconstruction_time,
                            "total_s": total_time,
                        }
                    ),
                    flush=True,
                )

        results[method] = {
            "encode": summarize(encode_times),
            "latent_model": summarize(latent_times),
            "reconstruction": summarize(reconstruction_times),
            "total": summarize(total_times),
        }
        del reduction_model, latent_predictor, trunks
        clear_cuda()

    summary = {
        "device": str(device),
        "data_path": str(args.data_path),
        "n_frames": int(fields.shape[0]),
        "n_windows": int(n_windows),
        "history": HISTORY,
        "future_steps": FUTURE_STEPS,
        "slices_per_encode": int(n_windows * HISTORY),
        "slices_per_reconstruction": int(n_windows * FUTURE_STEPS),
        "slice_batch_size": args.slice_batch_size,
        "window_batch_size": args.window_batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_scope": "HDF5 loading and fixed trunk preparation are excluded from the reported stage times.",
        "stage_definition": {
            "encode": "all input history field slices are encoded: n_windows * 10 slices.",
            "latent_model": "latent history windows predict the next 10 latent slices.",
            "reconstruction": "all predicted future latent slices are reconstructed: n_windows * 10 slices.",
        },
        "results": results,
    }
    json_path = args.output_dir / "cylinder_temp_stage_timing.json"
    csv_path = args.output_dir / "cylinder_temp_stage_timing.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(csv_path, summary)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
