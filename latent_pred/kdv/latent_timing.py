from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from latent import FNO


THIS_FILE = Path(__file__).resolve()
KDV_DIR = THIS_FILE.parent
PROJECT_ROOT = KDV_DIR.parents[1]
REDUCTION_DIR = PROJECT_ROOT / "reduction"
DEFAULT_DATA_PATH = REDUCTION_DIR / "data" / "data_kdv.mat"
DEFAULT_OUTPUT_DIR = KDV_DIR / "figures" / "training_timing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Timing variant of latent.py for KDV OAE/FOAE training.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reduction-model", default="DPN_d50")
    parser.add_argument("--reduction-model-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--pi-weight", type=float, default=1e-3)
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


def load_reduction_model(args: argparse.Namespace, device: torch.device):
    sys.path.insert(0, str(REDUCTION_DIR))
    import nn_kdv  # noqa: WPS433

    sys.modules["nn"] = nn_kdv
    globals()["DPN"] = nn_kdv.DPN
    globals()["FNO1"] = nn_kdv.FNO1
    globals()["FNO2"] = nn_kdv.FNO2
    globals()["ResidualBlock"] = nn_kdv.ResidualBlock
    globals()["SpectralConv1d"] = nn_kdv.SpectralConv1d

    path = args.reduction_model_path or (REDUCTION_DIR / "models" / f"{args.reduction_model}.pth")
    if not path.exists() and args.reduction_model == "DPN_d50":
        path = REDUCTION_DIR / "models" / "OAE_d50_kdv.pth"
    if not path.exists() and args.reduction_model == "DPN_d50_fno":
        path = REDUCTION_DIR / "models" / "FOAE_d50_kdv.pth"
    model = torch_load(path, device).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, path


def load_full_data(path: Path, batch_size: int) -> torch.Tensor:
    data = sio.loadmat(path)
    if "u" not in data:
        raise KeyError(f"{path} does not contain variable 'u'.")
    full = np.asarray(data["u"][:batch_size], dtype=np.float32)
    if full.ndim != 3:
        raise ValueError(f"Expected KDV data shaped [batch, time, space], got {full.shape}.")
    return torch.from_numpy(full)


@torch.no_grad()
def encode_full_batch(reduction_model, full: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    n_cases, n_times, n_space = full.shape
    flat = full.reshape(n_cases * n_times, n_space).to(device)
    parts = []
    for start in range(0, flat.shape[0], batch_size):
        parts.append(reduction_model.encode(flat[start : start + batch_size]).detach().cpu())
    return torch.cat(parts, dim=0).reshape(n_cases, n_times, -1).to(device)


@torch.no_grad()
def compute_trunk(reduction_model, device: torch.device) -> torch.Tensor:
    return reduction_model.trunk(device).permute(1, 0).contiguous()


def derivative_x(trunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = 1.0 / 512
    ux = torch.zeros_like(trunk)
    uxxx = torch.zeros_like(trunk)
    ux[..., 1:-1] = (trunk[..., 2:] - trunk[..., :-2]) / (2 * dx)
    uxxx[..., 2:-2] = (
        -trunk[..., :-4] + 2 * trunk[..., 1:-3] - 2 * trunk[..., 3:-1] + trunk[..., 4:]
    ) / (2 * dx**3)
    return ux, uxxx


def derivative_t(out: torch.Tensor) -> torch.Tensor:
    dt = 5.0 / 501
    ut = torch.zeros_like(out)
    ut[:, 1:-1] = (out[:, 2:] - out[:, :-2]) / (2 * dt)
    return ut


def reconstruct(out: torch.Tensor, trunk: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bti,io->bto", out, trunk)


def reconstruction_backward_probe(device: torch.device, out: torch.Tensor, trunk: torch.Tensor) -> tuple[float, float]:
    z = out.detach().requires_grad_(True)
    forward_time, full = timed(device, lambda: reconstruct(z, trunk))
    grad_full = torch.ones_like(full)
    backward_time, _ = timed(
        device,
        lambda: torch.autograd.grad(full, z, grad_outputs=grad_full, retain_graph=False, create_graph=False)[0],
    )
    return forward_time, backward_time


def mean_std(values: list[float]) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "runs_s": [float(value) for value in values],
    }


def write_csv(path: Path, summary: dict[str, object]) -> None:
    rows = []
    for stage, stats in summary["timing"].items():
        rows.append({"stage": stage, "mean_s": stats["mean_s"], "std_s": stats["std_s"]})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "mean_s", "std_s"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size != 4:
        raise ValueError("Keep --batch-size 4 for the requested timing experiment.")
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reduction_model, reduction_path = load_reduction_model(args, device)
    full = load_full_data(args.data_path, args.batch_size).to(device)
    u_latent = encode_full_batch(reduction_model, full, device, args.batch_size)
    trunk = compute_trunk(reduction_model, device)

    n_cases, n_times, latent_dim = u_latent.shape
    train_grid = torch.arange(0, n_times, step=50, device=device)
    x = u_latent[:, 0].reshape(n_cases, 1, latent_dim, 1).repeat(1, n_times, 1, 1)
    y = u_latent[:, train_grid]

    model = FNO(int(latent_dim / 2 + 1), args.modes, args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    mse_loss = nn.MSELoss().to(device)

    total_times: list[float] = []
    output_forward_times: list[float] = []
    reconstruction_backward_times: list[float] = []
    rows = []
    for epoch in range(args.warmup + args.epochs):
        def train_step():
            optimizer.zero_grad(set_to_none=True)
            out = model(x.permute(0, 2, 1, 3)).permute(0, 2, 1)
            mse = mse_loss(out[:, train_grid], y)
            full_out = reconstruct(out, trunk)
            ut = derivative_t(full_out)[:, 1:-1, 2:-2]
            ux, uxxx = derivative_x(full_out)
            orig = full_out[:, 1:-1, 2:-2]
            ux = ux[:, 1:-1, 2:-2]
            uxxx = uxxx[:, 1:-1, 2:-2]
            physics = mse_loss(ut + 6 * orig * ux, -0.001 * uxxx)
            loss = mse + args.pi_weight * physics
            loss.backward()
            optimizer.step()
            return out.detach()

        total_time, out_detached = timed(device, train_step)
        output_forward_time, reconstruction_backward_time = reconstruction_backward_probe(device, out_detached, trunk)
        if epoch >= args.warmup:
            total_times.append(total_time)
            output_forward_times.append(output_forward_time)
            reconstruction_backward_times.append(reconstruction_backward_time)
            row = {
                "epoch": epoch - args.warmup,
                "total_train_step_s": total_time,
                "latent_to_full_forward_s": output_forward_time,
                "full_to_latent_reconstruction_backward_s": reconstruction_backward_time,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    summary = {
        "script": str(THIS_FILE),
        "source": "latent.py",
        "data_path": str(args.data_path),
        "reduction_model_path": str(reduction_path),
        "batch_size": args.batch_size,
        "n_times": int(n_times),
        "n_space": int(full.shape[-1]),
        "latent_dim": int(latent_dim),
        "train_grid": train_grid.detach().cpu().numpy().astype(int).tolist(),
        "timing_note": (
            "total_train_step_s is a normal one-batch training step. "
            "latent_to_full_forward_s and full_to_latent_reconstruction_backward_s are separate probes; "
            "the backward probe is only the trunk reconstruction VJP from full-field gradient to latent."
        ),
        "timing": {
            "total_train_step": mean_std(total_times),
            "latent_to_full_forward": mean_std(output_forward_times),
            "full_to_latent_reconstruction_backward": mean_std(reconstruction_backward_times),
        },
        "per_epoch": rows,
    }
    json_path = args.output_dir / f"kdv_{args.reduction_model}_latent_training_timing.json"
    csv_path = args.output_dir / f"kdv_{args.reduction_model}_latent_training_timing.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(csv_path, summary)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "timing": summary["timing"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
