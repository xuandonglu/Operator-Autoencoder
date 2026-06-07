from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


THIS_FILE = Path(__file__).resolve()
OAE_ROOT = THIS_FILE.parents[2]
DEFAULT_DATA_PATH = OAE_ROOT / "reduction" / "data" / "data_cylinder.h5df"
DEFAULT_OUTPUT_DIR = THIS_FILE.parent / "results"

METHODS = {
    "foae": "latent_DPN_d80_fno_cylinder",
    "oae": "latent_DPN_d80_cylinder",
    "ae": "latent_AE_d140_cylinder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cylinder latent 2D FNO: truth z[t:t+10] predicts the next z[t+10:t+20]."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=["foae", "oae", "ae"])
    parser.add_argument("--train-cases", type=int, default=40)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--modes-latent", type=int, default=36)
    parser.add_argument("--modes-time", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_normalizer(latent: np.ndarray, train_cases: list[int]) -> tuple[np.ndarray, np.ndarray]:
    block = latent[train_cases]
    mean = block.mean(axis=(0, 1), keepdims=False).astype(np.float32)
    std = block.std(axis=(0, 1), keepdims=False).astype(np.float32)
    return mean, np.maximum(std, 1e-6).astype(np.float32)


class HistoryToFutureDataset(Dataset):
    def __init__(
        self,
        latent: np.ndarray,
        case_indices: list[int],
        history: int,
        future_steps: int,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.latent = latent
        self.history = history
        self.future_steps = future_steps
        self.mean = mean
        self.std = std
        stop = latent.shape[1] - history - future_steps + 1
        if stop <= 0:
            raise ValueError(
                f"history={history}, future_steps={future_steps} are invalid for {latent.shape[1]} time points."
            )
        self.samples = [(case, start_t) for case in case_indices for start_t in range(stop)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        case, start_t = self.samples[idx]
        sequence = (self.latent[case, start_t : start_t + self.history + self.future_steps] - self.mean) / self.std
        x = sequence[: self.history]
        y = sequence[self.history :]
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_latent: int, modes_time: int) -> None:
        super().__init__()
        self.modes_latent = modes_latent
        self.modes_time = modes_time
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_latent, modes_time, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_latent, modes_time, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        usable_latent = min(self.modes_latent, x_ft.shape[-2])
        usable_time = min(self.modes_time, x_ft.shape[-1])
        out_ft = torch.zeros(
            x.shape[0], self.weights1.shape[1], x_ft.shape[-2], x_ft.shape[-1], dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :usable_latent, :usable_time] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :usable_latent, :usable_time],
            self.weights1[:, :, :usable_latent, :usable_time],
        )
        out_ft[:, :, -usable_latent:, :usable_time] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -usable_latent:, :usable_time],
            self.weights2[:, :, :usable_latent, :usable_time],
        )
        return torch.fft.irfft2(out_ft, s=x.shape[-2:], dim=(-2, -1))


class LatentHistoryToFutureFNO2d(nn.Module):
    def __init__(
        self,
        latent_size: int,
        history: int,
        future_steps: int,
        width: int,
        layers: int,
        modes_latent: int,
        modes_time: int,
    ) -> None:
        super().__init__()
        if not 1 <= modes_latent <= latent_size // 2 + 1:
            raise ValueError(f"modes_latent must be in [1, {latent_size // 2 + 1}].")
        if not 1 <= modes_time <= future_steps // 2 + 1:
            raise ValueError(f"modes_time must be in [1, {future_steps // 2 + 1}].")
        self.latent_size = latent_size
        self.history = history
        self.future_steps = future_steps
        self.width = width
        self.layers = layers
        self.modes_latent = modes_latent
        self.modes_time = modes_time
        self.fc0 = nn.Linear(history + 2, width)
        self.convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes_latent, modes_time) for _ in range(layers)]
        )
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        latent_grid = torch.linspace(0.0, 1.0, latent_size).reshape(1, latent_size, 1, 1)
        time_grid = torch.linspace(0.0, 1.0, future_steps).reshape(1, 1, future_steps, 1)
        self.register_buffer("latent_grid", latent_grid, persistent=False)
        self.register_buffer("time_grid", time_grid, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        repeated = x.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, self.future_steps, -1)
        latent_grid = self.latent_grid.expand(batch_size, -1, self.future_steps, -1)
        time_grid = self.time_grid.expand(batch_size, self.latent_size, -1, -1)
        h = self.fc0(torch.cat([repeated, latent_grid, time_grid], dim=-1)).permute(0, 3, 1, 2)
        for conv, w in zip(self.convs, self.ws):
            h = F.gelu(conv(h) + w(h))
        h = h.permute(0, 2, 3, 1)
        return self.fc2(F.gelu(self.fc1(h))).squeeze(-1).permute(0, 2, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    latent: np.ndarray,
    case_indices: list[int],
    history: int,
    future_steps: int,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | list[float]]:
    model.eval()
    dataset = HistoryToFutureDataset(latent, case_indices, history, future_steps, mean, std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    mean_t = torch.from_numpy(mean.reshape(1, 1, -1)).to(device)
    std_t = torch.from_numpy(std.reshape(1, 1, -1)).to(device)
    norm_sse = 0.0
    latent_sse = 0.0
    count = 0
    step_sse = torch.zeros(future_steps, dtype=torch.float64)
    step_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        norm_sse += float(torch.square(pred - y).sum().cpu())
        pred_latent = pred * std_t + mean_t
        true_latent = y * std_t + mean_t
        diff = pred_latent - true_latent
        latent_sse += float(torch.square(diff).sum().cpu())
        step_sse += torch.square(diff).sum(dim=(0, 2)).detach().cpu().double()
        step_count += diff.shape[0] * diff.shape[2]
        count += pred.numel()
    per_step = (step_sse / max(1, step_count)).numpy().astype(float).tolist()
    return {
        "mse_norm": norm_sse / max(1, count),
        "mse_latent": latent_sse / max(1, count),
        "first_step_mse_latent": per_step[0],
        "last_step_mse_latent": per_step[-1],
        "per_step_mse_latent": per_step,
    }


def save_checkpoint(
    path: Path,
    model: LatentHistoryToFutureFNO2d,
    mean: np.ndarray,
    std: np.ndarray,
    method: str,
    epoch: int,
    metrics: dict[str, object],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "method": method,
            "latent_key": METHODS[method],
            "epoch": epoch,
            "metrics": metrics,
            "model_config": {
                "latent_size": model.latent_size,
                "history": model.history,
                "future_steps": model.future_steps,
                "width": model.width,
                "layers": model.layers,
                "modes_latent": model.modes_latent,
                "modes_time": model.modes_time,
            },
            "train_config": jsonable_args(args),
        },
        path,
    )


def train_one(args: argparse.Namespace, method: str, device: torch.device) -> dict[str, object]:
    with h5py.File(args.data_path, "r") as handle:
        latent = np.asarray(handle[METHODS[method]][:], dtype=np.float32)
    n_cases, n_times, latent_size = latent.shape
    train_cases = list(range(args.train_cases))
    test_cases = list(range(args.train_cases, n_cases))
    mean, std = compute_normalizer(latent, train_cases)
    train_dataset = HistoryToFutureDataset(latent, train_cases, args.history, args.future_steps, mean, std)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda"
    )
    model = LatentHistoryToFutureFNO2d(
        latent_size,
        args.history,
        args.future_steps,
        args.width,
        args.layers,
        args.modes_latent,
        args.modes_time,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    method_dir = args.output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "method": method,
        "latent_key": METHODS[method],
        "latent_shape": list(latent.shape),
        "mapping": f"z[t:t+{args.history}] -> z[t+{args.history}:t+{args.history + args.future_steps}]",
        "train_cases_0_based": train_cases,
        "test_cases_0_based": test_cases,
        "train_pairs": len(train_dataset),
        "test_pairs": len(test_cases) * (n_times - args.history - args.future_steps + 1),
        "model": {
            "type": "LatentHistoryToFutureFNO2d",
            "latent_size": latent_size,
            "history": args.history,
            "future_steps": args.future_steps,
            "width": args.width,
            "layers": args.layers,
            "modes_latent": args.modes_latent,
            "modes_time": args.modes_time,
        },
        "args": jsonable_args(args),
    }
    (method_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_train = math.inf
    best_epoch = -1
    best_observed_test = math.inf
    best_observed_test_epoch = -1
    rows: list[dict[str, object]] = []
    start_all = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        start_epoch = time.perf_counter()
        total_sse = 0.0
        count = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            total_sse += float(torch.square(pred.detach() - y).sum().cpu())
            count += pred.numel()
        scheduler.step()
        row: dict[str, object] = {
            "method": method,
            "epoch": epoch,
            "train_mse_norm": total_sse / max(1, count),
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - start_epoch,
        }
        if epoch == 0 or (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            metrics = evaluate(
                model, latent, test_cases, args.history, args.future_steps, mean, std, device, args.batch_size
            )
            row.update({f"test_{key}": value for key, value in metrics.items() if not isinstance(value, list)})
            row["test_per_step_mse_latent"] = json.dumps(metrics["per_step_mse_latent"])
            if float(metrics["mse_norm"]) < best_observed_test:
                best_observed_test = float(metrics["mse_norm"])
                best_observed_test_epoch = epoch
                save_checkpoint(method_dir / "best_observed_test.pt", model, mean, std, method, epoch, row, args)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if row["train_mse_norm"] < best_train:
            best_train = float(row["train_mse_norm"])
            best_epoch = epoch
            save_checkpoint(method_dir / "best_train.pt", model, mean, std, method, epoch, row, args)
    save_checkpoint(method_dir / "last.pt", model, mean, std, method, args.epochs - 1, rows[-1], args)
    write_csv(method_dir / "history.csv", rows)
    checkpoint = torch.load(method_dir / "best_train.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    best_train_test = evaluate(model, latent, test_cases, args.history, args.future_steps, mean, std, device, args.batch_size)
    summary = {
        "method": method,
        "latent_key": METHODS[method],
        "latent_shape": list(latent.shape),
        "mapping": config["mapping"],
        "train_cases_0_based": train_cases,
        "test_cases_0_based": test_cases,
        "train_pairs": len(train_dataset),
        "test_pairs": len(test_cases) * (n_times - args.history - args.future_steps + 1),
        "best_epoch_by_train": best_epoch,
        "best_train_mse_norm": best_train,
        "best_train_checkpoint_test": best_train_test,
        "best_observed_test_epoch": best_observed_test_epoch,
        "best_observed_test_mse_norm": best_observed_test,
        "training_seconds": time.perf_counter() - start_all,
    }
    (method_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = jsonable_args(args)
    config["device_resolved"] = str(device)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summaries = []
    for method in args.methods:
        set_seed(args.seed)
        summaries.append(train_one(args, method, device))
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
