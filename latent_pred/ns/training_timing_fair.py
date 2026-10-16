from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch
import torch.nn as nn

import latent_ae_timing as ae_timing
import latent_timing as dpn_timing


THIS_FILE = Path(__file__).resolve()
NS_DIR = THIS_FILE.parent
DEFAULT_OUTPUT_DIR = NS_DIR / "figures" / "training_timing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified NS training timing comparison.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
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


def reconstruction_backward_once(device: torch.device, reconstruct, out: torch.Tensor) -> float:
    z = out.detach().requires_grad_(True)
    full = reconstruct(z)
    grad_full = torch.ones_like(full)
    sync(device)
    start = perf_counter()
    torch.autograd.grad(full, z, grad_outputs=grad_full, retain_graph=False, create_graph=False)[0]
    sync(device)
    return perf_counter() - start


def measure_reconstruction_backward(device: torch.device, reconstruct, out: torch.Tensor, warmup: int, steps: int):
    values = []
    for idx in range(warmup + steps):
        value = reconstruction_backward_once(device, reconstruct, out)
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


def setup_dpn_method(
    label: str,
    reduction_model_name: str,
    pi_weight: float,
    data_path: Path,
    full_key: str,
    force: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
):
    reduction_args = argparse.Namespace(reduction_model=reduction_model_name, reduction_model_path=None)
    reduction_model, reduction_path = dpn_timing.load_reduction_model(reduction_args, device)
    latent_key = f"latent_{reduction_model_name}"
    trunk_key = f"trunk_{reduction_model_name}"
    with h5py.File(data_path, "r") as handle:
        latent_np = dpn_timing.load_or_compute_latent(
            handle, latent_key, full_key, reduction_model, device, args.batch_size
        )
        trunk = dpn_timing.load_or_compute_trunk(handle, trunk_key, reduction_model, device)

    latent = torch.from_numpy(latent_np).float().to(device)
    batch_size, n_times, latent_dim = latent.shape
    train_grid = torch.arange(0, n_times, step=9, device=device)
    x = latent[:, 0].reshape(batch_size, 1, latent_dim, 1).repeat(1, n_times, 1, 1)
    y = latent[:, train_grid]
    model = make_latent_model(latent_dim, args.modes, args.width, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    mse_loss = nn.MSELoss().to(device)
    basis_ux, basis_wx, basis_uy, basis_wy, basis_d2w = dpn_timing.trunk_derivative(trunk)

    def latent_forward():
        return model(x.permute(0, 2, 1, 3)).permute(0, 2, 1)

    def reconstruct(out: torch.Tensor):
        return dpn_timing.reconstruct(out, trunk)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        out = latent_forward()
        latent_mse = mse_loss(out[:, train_grid], y)
        zt = dpn_timing.derivative_t(out)
        wt = torch.einsum("bti,ixy->btxy", zt, trunk.permute(0, 2, 1))
        out_mid = out[:, 1:-1]
        ux = torch.einsum("bti,ixy->btxy", out_mid, basis_ux)
        uy = torch.einsum("bti,ixy->btxy", out_mid, basis_uy)
        wx = torch.einsum("bti,ixy->btxy", out_mid, basis_wx)
        wy = torch.einsum("bti,ixy->btxy", out_mid, basis_wy)
        d2w = torch.einsum("bti,ixy->btxy", out_mid, basis_d2w)
        physics = mse_loss(wt + ux * wx + uy * wy, d2w + force)
        loss = latent_mse + pi_weight * physics
        loss.backward()
        optimizer.step()

    return {
        "label": label,
        "reduction_model": reduction_model_name,
        "reduction_model_path": str(reduction_path),
        "pi_weight": pi_weight,
        "latent": latent,
        "train_grid": train_grid,
        "latent_forward": latent_forward,
        "reconstruct": reconstruct,
        "train_step": train_step,
        "residual_path": "basis/trunk derivatives, matching latent_timing.py",
    }


def setup_ae_method(
    data_path: Path,
    full_key: str,
    force: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
):
    ae_args = argparse.Namespace(reduction_model="fno_d150_m12_w1", reduction_model_path=None)
    ae_model, ae_path = ae_timing.load_ae(ae_args, device)
    latent_key = "u_latent_fno_d150_m12_w1"
    with h5py.File(data_path, "r") as handle:
        latent_np = ae_timing.load_or_compute_latent(
            handle, latent_key, full_key, ae_model, device, args.batch_size
        )

    latent = torch.from_numpy(latent_np).float().to(device)
    batch_size, n_times, latent_dim = latent.shape
    train_grid = torch.arange(0, n_times, step=9, device=device)
    x = latent[:, 0].reshape(batch_size, 1, latent_dim, 1).repeat(1, n_times, 1, 1)
    y = latent[:, train_grid]
    model = make_latent_model(latent_dim, args.modes, args.width, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    mse_loss = nn.MSELoss().to(device)
    pi_weight = 0.5
    size_x = size_y = 64

    def latent_forward():
        return model(x.permute(0, 2, 1, 3)).permute(0, 2, 1)

    def reconstruct(out: torch.Tensor):
        return ae_timing.reconstruct(ae_model, out, size_x, size_y)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        out = latent_forward()
        latent_mse = mse_loss(out[:, train_grid], y)
        full = reconstruct(out)
        lhs, rhs = ae_timing.derivative(full, force)
        physics = mse_loss(lhs[:, 1:-1], rhs[:, 1:-1])
        loss = latent_mse + pi_weight * physics
        loss.backward()
        optimizer.step()

    return {
        "label": "AE",
        "reduction_model": "fno_d150_m12_w1",
        "reduction_model_path": str(ae_path),
        "pi_weight": pi_weight,
        "latent": latent,
        "train_grid": train_grid,
        "latent_forward": latent_forward,
        "reconstruct": reconstruct,
        "train_step": train_step,
        "residual_path": "decoded full-field FFT derivatives, matching latent_ae_timing.py",
    }


@torch.no_grad()
def fixed_latent_output(method: dict[str, object]) -> torch.Tensor:
    return method["latent_forward"]().detach()


def write_csv(path: Path, summary: dict[str, object]) -> None:
    fieldnames = ["model", "stage", "mean_s", "median_s", "std_s", "p10_s", "p90_s"]
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
        raise ValueError("Keep --batch-size 4 for this NS timing batch.")
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)
    data_path = dpn_timing.resolve_data_path(args.data_path)
    with h5py.File(data_path, "r") as handle:
        full_key = dpn_timing.find_full_key(handle, None)
        force = dpn_timing.force_tensor(handle, device)

    methods = [
        setup_dpn_method("OAE", "DPN_d150", 1.0, data_path, full_key, force, args, device),
        setup_dpn_method("FOAE", "DPN_d150_fno", 1.0, data_path, full_key, force, args, device),
        setup_ae_method(data_path, full_key, force, args, device),
    ]

    model_summaries = []
    for method in methods:
        print(f"Timing {method['label']}...", flush=True)
        total = measure(device, method["train_step"], args.warmup, args.steps)
        out = fixed_latent_output(method)
        recon_forward = measure(
            device,
            lambda method=method, out=out: method["reconstruct"](out.detach().requires_grad_(True)),
            args.warmup,
            args.steps,
        )
        recon_backward = measure_reconstruction_backward(
            device, method["reconstruct"], out, args.warmup, args.steps
        )
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
            "residual_path": method["residual_path"],
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
        "data_path": str(data_path),
        "full_key": full_key,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "timing_note": (
            "All models are timed in one process on the same NS batch. Reconstruction probes use a fixed latent "
            "output and do not include latent FNO forward. The DPN methods intentionally keep the original "
            "basis/trunk residual path, while AE keeps the decoded full-field FFT residual path."
        ),
        "models": model_summaries,
    }
    json_path = args.output_dir / "ns_fair_training_timing.json"
    csv_path = args.output_dir / "ns_fair_training_timing.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(csv_path, summary)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
