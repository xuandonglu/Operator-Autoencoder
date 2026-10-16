from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn

from latent_ae import FNO as LatentFNO


THIS_FILE = Path(__file__).resolve()
NS_DIR = THIS_FILE.parent
PROJECT_ROOT = NS_DIR.parents[1]
REDUCTION_DIR = PROJECT_ROOT / "reduction"
DEFAULT_OUTPUT_DIR = NS_DIR / "figures" / "training_timing"
DEFAULT_CANDIDATE_DATA_PATHS = [
    REDUCTION_DIR / "data" / "data_ns.h5df",
    REDUCTION_DIR / "data" / "ns2d" / "ns_data5.h5df",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Timing variant of latent_ae.py for NS AE training.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reduction-model", default="fno_d150_m12_w1")
    parser.add_argument("--reduction-model-path", type=Path, default=None)
    parser.add_argument("--latent-key", default=None)
    parser.add_argument("--full-key", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--pi-weight", type=float, default=0.5)
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


def resolve_data_path(path: Path | None) -> Path:
    if path is not None:
        return path
    for candidate in DEFAULT_CANDIDATE_DATA_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No NS HDF5 data file found. Pass --data-path, or generate ns_data5.h5df first."
    )


def find_full_key(handle: h5py.File, requested: str | None) -> str:
    if requested is not None:
        return requested
    candidates = [
        key
        for key in handle.keys()
        if key.startswith("ns_noise_") and key.endswith("_161_64_64") and len(handle[key].shape) == 4
    ]
    if not candidates:
        raise KeyError("Could not find an ns_noise_*_161_64_64 dataset.")
    return sorted(candidates)[0]


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_ae(args: argparse.Namespace, device: torch.device):
    sys.path.insert(0, str(REDUCTION_DIR))
    import nn_ae_ns  # noqa: WPS433

    sys.modules["nn"] = nn_ae_ns
    globals()["AE"] = nn_ae_ns.AE
    globals()["SpectralConv2d"] = nn_ae_ns.SpectralConv2d
    globals()["FNO"] = nn_ae_ns.FNO
    path = args.reduction_model_path or (REDUCTION_DIR / "models" / f"{args.reduction_model}.pth")
    if not path.exists() and args.reduction_model == "fno_d150_m12_w1":
        path = REDUCTION_DIR / "models" / "AE_d150_ns.pth"
    model = torch_load(path, device).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, path


@torch.no_grad()
def encode_current_batch(ae_model, full_batch: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    n_cases, n_times, size_x, size_y = full_batch.shape
    flat = torch.from_numpy(full_batch.reshape(n_cases * n_times, 1, size_x, size_y).astype(np.float32)).to(device)
    parts = []
    for start in range(0, flat.shape[0], batch_size):
        parts.append(ae_model.encode(flat[start : start + batch_size]).detach().cpu())
    return torch.cat(parts, dim=0).numpy().reshape(n_cases, n_times, -1).astype(np.float32)


@torch.no_grad()
def load_or_compute_latent(
    handle: h5py.File,
    latent_key: str,
    full_key: str,
    ae_model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if latent_key in handle and handle[latent_key].shape[0] >= batch_size:
        return np.asarray(handle[latent_key][:batch_size], dtype=np.float32)
    full_batch = np.asarray(handle[full_key][:batch_size], dtype=np.float32)
    return encode_current_batch(ae_model, full_batch, device, batch_size)


def force_tensor(handle: h5py.File, device: torch.device) -> torch.Tensor:
    if "f" not in handle:
        raise KeyError("Dataset 'f' is required for the NS physics loss.")
    return torch.from_numpy(np.asarray(handle["f"], dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)


def derivative(full: torch.Tensor, force: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    u = full.permute(0, 1, 3, 2)
    n = u.shape[-1]
    nu = 0.001
    dt = 0.25
    w_t = torch.zeros_like(u)
    w_t[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dt)
    k = torch.fft.fftfreq(n, d=1.0 / n).to(u.device)
    k_x, k_y = torch.meshgrid(k, k, indexing="ij")
    lap = -(2 * torch.pi) ** 2 * (k_x**2 + k_y**2)
    lap[0, 0] = 1.0
    w_h = fft.fft2(u)
    psi_h = w_h / lap
    u_x = torch.real(fft.ifft2(1j * 2 * torch.pi * k_y * psi_h))
    u_y = torch.real(fft.ifft2(-1j * 2 * torch.pi * k_x * psi_h))
    w_x = torch.real(fft.ifft2(1j * 2 * torch.pi * k_x * w_h))
    w_y = torch.real(fft.ifft2(1j * 2 * torch.pi * k_y * w_h))
    d2w = nu * torch.real(fft.ifft2(lap * w_h))
    lhs = w_t + u_x * w_x + u_y * w_y
    rhs = d2w + force
    return lhs, rhs


def reconstruct(ae_model, out: torch.Tensor, size_x: int, size_y: int) -> torch.Tensor:
    batch_size, n_times, latent_dim = out.shape
    return ae_model.decode(out.reshape(batch_size * n_times, latent_dim)).reshape(batch_size, n_times, size_x, size_y)


def reconstruction_backward_probe(
    device: torch.device,
    ae_model,
    out: torch.Tensor,
    size_x: int,
    size_y: int,
) -> tuple[float, float]:
    z = out.detach().requires_grad_(True)
    forward_time, full = timed(device, lambda: reconstruct(ae_model, z, size_x, size_y))
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
    data_path = resolve_data_path(args.data_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ae_model, ae_path = load_ae(args, device)
    latent_key = args.latent_key or f"u_latent_{args.reduction_model}"

    with h5py.File(data_path, "r") as handle:
        full_key = find_full_key(handle, args.full_key)
        latent_np = load_or_compute_latent(handle, latent_key, full_key, ae_model, device, args.batch_size)
        force = force_tensor(handle, device)

    u_latent = torch.from_numpy(latent_np).float().to(device)
    n_cases, n_times, latent_dim = u_latent.shape
    train_grid = torch.arange(0, n_times, step=9, device=device)
    x = u_latent[:, 0].reshape(n_cases, 1, latent_dim, 1).repeat(1, n_times, 1, 1)
    y = u_latent[:, train_grid]
    size_x = size_y = 64

    model = LatentFNO(int(latent_dim / 2 + 1), args.modes, args.width).to(device)
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
            full = reconstruct(ae_model, out, size_x, size_y)
            lhs, rhs = derivative(full, force)
            physics = mse_loss(lhs[:, 1:-1], rhs[:, 1:-1])
            loss = mse + args.pi_weight * physics
            loss.backward()
            optimizer.step()
            return out.detach()

        total_time, out_detached = timed(device, train_step)
        output_forward_time, reconstruction_backward_time = reconstruction_backward_probe(
            device, ae_model, out_detached, size_x, size_y
        )
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
        "source": "latent_ae.py",
        "data_path": str(data_path),
        "full_key": full_key,
        "latent_key": latent_key,
        "ae_model_path": str(ae_path),
        "batch_size": args.batch_size,
        "n_times": int(n_times),
        "latent_dim": int(latent_dim),
        "train_grid": train_grid.detach().cpu().numpy().astype(int).tolist(),
        "timing_note": (
            "total_train_step_s is a normal one-batch training step. "
            "latent_to_full_forward_s and full_to_latent_reconstruction_backward_s are separate probes; "
            "the backward probe is only the decoder VJP from full-field gradient to latent."
        ),
        "timing": {
            "total_train_step": mean_std(total_times),
            "latent_to_full_forward": mean_std(output_forward_times),
            "full_to_latent_reconstruction_backward": mean_std(reconstruction_backward_times),
        },
        "per_epoch": rows,
    }
    json_path = args.output_dir / f"ns_{args.reduction_model}_ae_training_timing.json"
    csv_path = args.output_dir / f"ns_{args.reduction_model}_ae_training_timing.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(csv_path, summary)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "timing": summary["timing"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
