from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn

import latent_ae_timing as ae_timing
import latent_timing as dpn_timing


THIS_FILE = Path(__file__).resolve()
KDV_DIR = THIS_FILE.parent
PROJECT_ROOT = KDV_DIR.parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "reduction" / "data" / "data_kdv.mat"
DEFAULT_OUTPUT_DIR = KDV_DIR / "figures" / "training_timing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair KDV training timing comparison for OAE, FOAE, and AE.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(device: torch.device, fn, warmup: int, steps: int) -> dict[str, object]:
    values = []
    for idx in range(warmup + steps):
        sync(device)
        start = perf_counter()
        fn()
        sync(device)
        if idx >= warmup:
            values.append(perf_counter() - start)
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "std_s": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p10_s": float(np.percentile(arr, 10)),
        "p90_s": float(np.percentile(arr, 90)),
        "runs_s": [float(value) for value in values],
    }


def make_latent_model(latent_dim: int, modes: int, width: int, device: torch.device):
    torch.manual_seed(123)
    return dpn_timing.FNO(int(latent_dim / 2 + 1), modes, width).to(device)


def setup_dpn_method(
    label: str,
    reduction_model_name: str,
    pi_weight: float,
    full: torch.Tensor,
    train_grid: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
):
    reduction_args = argparse.Namespace(
        reduction_model=reduction_model_name,
        reduction_model_path=None,
        batch_size=args.batch_size,
    )
    reduction_model, reduction_path = dpn_timing.load_reduction_model(reduction_args, device)
    latent = dpn_timing.encode_full_batch(reduction_model, full, device, args.batch_size)
    trunk = dpn_timing.compute_trunk(reduction_model, device)
    batch_size, n_times, latent_dim = latent.shape
    x = latent[:, 0].reshape(batch_size, 1, latent_dim, 1).repeat(1, n_times, 1, 1)
    y = latent[:, train_grid]
    model = make_latent_model(latent_dim, args.modes, args.width, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    mse_loss = nn.MSELoss().to(device)

    def latent_forward():
        return model(x.permute(0, 2, 1, 3)).permute(0, 2, 1)

    def reconstruct(out: torch.Tensor):
        return dpn_timing.reconstruct(out, trunk)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        out = latent_forward()
        latent_mse = mse_loss(out[:, train_grid], y)
        full_out = reconstruct(out)
        ut = dpn_timing.derivative_t(full_out)[:, 1:-1, 2:-2]
        ux, uxxx = dpn_timing.derivative_x(full_out)
        orig = full_out[:, 1:-1, 2:-2]
        physics = mse_loss(ut + 6 * orig * ux[:, 1:-1, 2:-2], -0.001 * uxxx[:, 1:-1, 2:-2])
        loss = latent_mse + pi_weight * physics
        loss.backward()
        optimizer.step()

    return {
        "label": label,
        "reduction_model": reduction_model_name,
        "reduction_model_path": str(reduction_path),
        "pi_weight": pi_weight,
        "latent": latent,
        "latent_forward": latent_forward,
        "reconstruct": reconstruct,
        "train_step": train_step,
    }


def setup_ae_method(
    full: torch.Tensor,
    train_grid: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
):
    ae_args = argparse.Namespace(
        reduction_model="fno_d50_m20_w1",
        reduction_model_path=None,
        batch_size=args.batch_size,
    )
    ae_model, ae_path = ae_timing.load_ae(ae_args, device)
    latent = ae_timing.encode_full_batch(ae_model, full, device, args.batch_size)
    batch_size, n_times, latent_dim = latent.shape
    n_space = full.shape[-1]
    x = latent[:, 0].reshape(batch_size, 1, latent_dim, 1).repeat(1, n_times, 1, 1)
    y = latent[:, train_grid]
    model = make_latent_model(latent_dim, args.modes, args.width, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    mse_loss = nn.MSELoss().to(device)
    pi_weight = 10.0

    def latent_forward():
        return model(x.permute(0, 2, 1, 3)).permute(0, 2, 1)

    def reconstruct(out: torch.Tensor):
        return ae_timing.reconstruct(ae_model, out, n_space)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        out = latent_forward()
        latent_mse = mse_loss(out[:, train_grid], y)
        full_out = reconstruct(out)
        ut = ae_timing.derivative_t(full_out)[:, 1:-1, 2:-2]
        ux, uxxx = ae_timing.derivative_x(full_out)
        orig = full_out[:, 1:-1, 2:-2]
        physics = mse_loss(ut + 6 * orig * ux[:, 1:-1, 2:-2], -0.001 * uxxx[:, 1:-1, 2:-2])
        loss = latent_mse + pi_weight * physics
        loss.backward()
        optimizer.step()

    return {
        "label": "AE",
        "reduction_model": "fno_d50_m20_w1",
        "reduction_model_path": str(ae_path),
        "pi_weight": pi_weight,
        "latent": latent,
        "latent_forward": latent_forward,
        "reconstruct": reconstruct,
        "train_step": train_step,
    }


@torch.no_grad()
def fixed_latent_output(method: dict[str, object]) -> torch.Tensor:
    return method["latent_forward"]().detach()


def reconstruction_forward_once(method: dict[str, object], out: torch.Tensor):
    z = out.detach().requires_grad_(True)
    method["reconstruct"](z)


def reconstruction_backward_once(device: torch.device, method: dict[str, object], out: torch.Tensor) -> float:
    z = out.detach().requires_grad_(True)
    full = method["reconstruct"](z)
    grad_full = torch.ones_like(full)
    sync(device)
    start = perf_counter()
    torch.autograd.grad(full, z, grad_outputs=grad_full, retain_graph=False, create_graph=False)[0]
    sync(device)
    return perf_counter() - start


def measure_reconstruction_backward(device: torch.device, method: dict[str, object], out: torch.Tensor, warmup: int, steps: int):
    values = []
    for idx in range(warmup + steps):
        value = reconstruction_backward_once(device, method, out)
        if idx >= warmup:
            values.append(value)
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "std_s": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p10_s": float(np.percentile(arr, 10)),
        "p90_s": float(np.percentile(arr, 90)),
        "runs_s": [float(value) for value in values],
    }


def write_csv(path: Path, summary: dict[str, object]) -> None:
    fieldnames = [
        "model",
        "stage",
        "mean_s",
        "median_s",
        "std_s",
        "p10_s",
        "p90_s",
    ]
    rows = []
    for item in summary["models"]:
        for stage, stats in item["timing"].items():
            rows.append(
                {
                    "model": item["label"],
                    "stage": stage,
                    "mean_s": stats["mean_s"],
                    "median_s": stats["median_s"],
                    "std_s": stats["std_s"],
                    "p10_s": stats["p10_s"],
                    "p90_s": stats["p90_s"],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size != 4:
        raise ValueError("Keep --batch-size 4 for this KDV timing batch.")
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)
    full = dpn_timing.load_full_data(args.data_path, args.batch_size).to(device)
    _, n_times, _ = full.shape
    train_grid = torch.arange(0, n_times, step=50, device=device)

    methods = [
        setup_dpn_method("OAE", "DPN_d50", 1e-3, full, train_grid, args, device),
        setup_dpn_method("FOAE", "DPN_d50_fno", 1e-1, full, train_grid, args, device),
        setup_ae_method(full, train_grid, args, device),
    ]

    model_summaries = []
    for method in methods:
        print(f"Timing {method['label']}...", flush=True)
        total = measure(device, method["train_step"], args.warmup, args.steps)
        out = fixed_latent_output(method)
        recon_forward = measure(
            device,
            lambda method=method, out=out: reconstruction_forward_once(method, out),
            args.warmup,
            args.steps,
        )
        recon_backward = measure_reconstruction_backward(device, method, out, args.warmup, args.steps)
        timing = {
            "total_train_step": total,
            "latent_to_full_forward": recon_forward,
            "full_to_latent_reconstruction_backward": recon_backward,
        }
        model_summary = {
            "label": method["label"],
            "reduction_model": method["reduction_model"],
            "reduction_model_path": method["reduction_model_path"],
            "pi_weight": method["pi_weight"],
            "latent_shape": list(method["latent"].shape),
            "timing": timing,
        }
        model_summaries.append(model_summary)
        compact = {
            "model": method["label"],
            "total_median_ms": total["median_s"] * 1000,
            "recon_forward_median_ms": recon_forward["median_s"] * 1000,
            "recon_backward_median_ms": recon_backward["median_s"] * 1000,
        }
        print(json.dumps(compact), flush=True)

    summary = {
        "script": str(THIS_FILE),
        "data_path": str(args.data_path),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "train_grid": train_grid.detach().cpu().numpy().astype(int).tolist(),
        "timing_note": (
            "All models are timed in one process on the same KDV batch, with the same latent FNO shape and "
            "the same full-field KDV residual. Reconstruction probes use a fixed latent output and do not "
            "include latent FNO forward."
        ),
        "models": model_summaries,
    }
    json_path = args.output_dir / "kdv_fair_training_timing.json"
    csv_path = args.output_dir / "kdv_fair_training_timing.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(csv_path, summary)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
