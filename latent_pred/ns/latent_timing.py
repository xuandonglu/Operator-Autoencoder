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
import torch.nn as nn
import torch.fft as fft

from latent import FNO


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
    parser = argparse.ArgumentParser(description="Timing variant of latent.py for NS OAE/FOAE training.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reduction-model", default="DPN_d150")
    parser.add_argument("--reduction-model-path", type=Path, default=None)
    parser.add_argument("--latent-key", default=None)
    parser.add_argument("--trunk-key", default=None)
    parser.add_argument("--full-key", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--pi-weight", type=float, default=1.0)
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


def load_reduction_model(args: argparse.Namespace, device: torch.device):
    sys.path.insert(0, str(REDUCTION_DIR))
    import nn_ns  # noqa: WPS433

    sys.modules["nn"] = nn_ns
    globals()["DPN"] = nn_ns.DPN
    globals()["FNO1"] = nn_ns.FNO1
    globals()["FNO2"] = nn_ns.FNO2
    globals()["ResidualBlock"] = nn_ns.ResidualBlock
    globals()["SpectralConv2d"] = nn_ns.SpectralConv2d
    path = args.reduction_model_path or (REDUCTION_DIR / "models" / f"{args.reduction_model}.pth")
    if not path.exists() and args.reduction_model == "DPN_d150":
        path = REDUCTION_DIR / "models" / "OAE_d150_ns.pth"
    if not path.exists() and args.reduction_model == "DPN_d150_fno":
        path = REDUCTION_DIR / "models" / "FOAE_d150_ns.pth"
    model = torch_load(path, device).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, path


@torch.no_grad()
def encode_current_batch(model, full_batch: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    n_cases, n_times, size_x, size_y = full_batch.shape
    flat = torch.from_numpy(full_batch.reshape(n_cases * n_times, size_x, size_y).astype(np.float32)).to(device)
    parts = []
    for start in range(0, flat.shape[0], batch_size):
        parts.append(model.encode(flat[start : start + batch_size]).detach().cpu())
    return torch.cat(parts, dim=0).numpy().reshape(n_cases, n_times, -1).astype(np.float32)


@torch.no_grad()
def load_or_compute_latent(
    handle: h5py.File,
    latent_key: str,
    full_key: str,
    reduction_model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if latent_key in handle and handle[latent_key].shape[0] >= batch_size:
        return np.asarray(handle[latent_key][:batch_size], dtype=np.float32)
    full_batch = np.asarray(handle[full_key][:batch_size], dtype=np.float32)
    return encode_current_batch(reduction_model, full_batch, device, batch_size)


@torch.no_grad()
def load_or_compute_trunk(handle: h5py.File, trunk_key: str, reduction_model, device: torch.device) -> torch.Tensor:
    if trunk_key in handle:
        return torch.from_numpy(np.asarray(handle[trunk_key], dtype=np.float32)).to(device)
    z = torch.zeros(1, reduction_model.latent_dim, device=device)
    return reduction_model.trunk(z).permute(2, 0, 1).contiguous()


def force_tensor(handle: h5py.File, device: torch.device) -> torch.Tensor:
    if "f" not in handle:
        raise KeyError("Dataset 'f' is required for the NS physics loss.")
    return torch.from_numpy(np.asarray(handle["f"], dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)


def derivative_t(out: torch.Tensor) -> torch.Tensor:
    dt = 0.25
    return (out[:, 2:] - out[:, :-2]) / (2 * dt)


def trunk_derivative(trunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    basis = trunk.permute(0, 2, 1)
    n = basis.shape[-1]
    nu = 0.001
    k = torch.fft.fftfreq(n, d=1.0 / n).to(basis.device)
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    lap = -(2 * torch.pi) ** 2 * (kx**2 + ky**2)
    lap[0, 0] = 1.0
    wh = fft.fft2(basis)
    psi_h = wh / lap
    ux = torch.real(fft.ifft2(1j * 2 * torch.pi * ky * psi_h))
    uy = torch.real(fft.ifft2(-1j * 2 * torch.pi * kx * psi_h))
    wx = torch.real(fft.ifft2(1j * 2 * torch.pi * kx * wh))
    wy = torch.real(fft.ifft2(1j * 2 * torch.pi * ky * wh))
    d2w = nu * torch.real(fft.ifft2(lap * wh))
    return ux, wx, uy, wy, d2w


def reconstruct(out: torch.Tensor, trunk: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bti,ixy->btxy", out, trunk)


def reconstruction_backward_probe(
    device: torch.device,
    out: torch.Tensor,
    trunk: torch.Tensor,
) -> tuple[float, float]:
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
    data_path = resolve_data_path(args.data_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reduction_model, reduction_path = load_reduction_model(args, device)
    latent_key = args.latent_key or f"latent_{args.reduction_model}"
    trunk_key = args.trunk_key or f"trunk_{args.reduction_model}"

    with h5py.File(data_path, "r") as handle:
        full_key = find_full_key(handle, args.full_key)
        latent_np = load_or_compute_latent(handle, latent_key, full_key, reduction_model, device, args.batch_size)
        trunk = load_or_compute_trunk(handle, trunk_key, reduction_model, device)
        force = force_tensor(handle, device)

    u_latent = torch.from_numpy(latent_np).float().to(device)
    n_cases, n_times, latent_dim = u_latent.shape
    train_grid = torch.arange(0, n_times, step=9, device=device)
    x = u_latent[:, 0].reshape(n_cases, 1, latent_dim, 1).repeat(1, n_times, 1, 1)
    y = u_latent[:, train_grid]

    model = FNO(76, args.modes, args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    mse_loss = nn.MSELoss().to(device)
    basis_ux, basis_wx, basis_uy, basis_wy, basis_d2w = trunk_derivative(trunk)

    total_times: list[float] = []
    output_forward_times: list[float] = []
    reconstruction_backward_times: list[float] = []
    rows = []
    for epoch in range(args.warmup + args.epochs):
        def train_step():
            optimizer.zero_grad(set_to_none=True)
            out = model(x.permute(0, 2, 1, 3)).permute(0, 2, 1)
            mse = mse_loss(out[:, train_grid], y)
            zt = derivative_t(out)
            wt = torch.einsum("bti,ixy->btxy", zt, trunk.permute(0, 2, 1))
            out_mid = out[:, 1:-1]
            ux = torch.einsum("bti,ixy->btxy", out_mid, basis_ux)
            uy = torch.einsum("bti,ixy->btxy", out_mid, basis_uy)
            wx = torch.einsum("bti,ixy->btxy", out_mid, basis_wx)
            wy = torch.einsum("bti,ixy->btxy", out_mid, basis_wy)
            d2w = torch.einsum("bti,ixy->btxy", out_mid, basis_d2w)
            physics = mse_loss(wt + ux * wx + uy * wy, d2w + force)
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
        "data_path": str(data_path),
        "full_key": full_key,
        "latent_key": latent_key,
        "trunk_key": trunk_key,
        "reduction_model_path": str(reduction_path),
        "batch_size": args.batch_size,
        "n_times": int(n_times),
        "latent_dim": int(latent_dim),
        "train_grid": train_grid.detach().cpu().numpy().astype(int).tolist(),
        "timing_note": (
            "total_train_step_s is a normal one-batch training step. "
            "latent_to_full_forward_s and full_to_latent_reconstruction_backward_s are separate probes; "
            "the backward probe is only the reconstruction VJP from full-field gradient to latent."
        ),
        "timing": {
            "total_train_step": mean_std(total_times),
            "latent_to_full_forward": mean_std(output_forward_times),
            "full_to_latent_reconstruction_backward": mean_std(reconstruction_backward_times),
        },
        "per_epoch": rows,
    }
    json_path = args.output_dir / f"ns_{args.reduction_model}_latent_training_timing.json"
    csv_path = args.output_dir / f"ns_{args.reduction_model}_latent_training_timing.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(csv_path, summary)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "timing": summary["timing"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
