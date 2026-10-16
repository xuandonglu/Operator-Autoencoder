from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


THIS_FILE = Path(__file__).resolve()
WAVE_DIR = THIS_FILE.parent
PROJECT_ROOT = WAVE_DIR.parents[1]
REDUCTION_DIR = PROJECT_ROOT / "reduction"
DEFAULT_DATA_PATH = REDUCTION_DIR / "data" / "data_wave.h5df"
DEFAULT_OUTPUT_DIR = WAVE_DIR / "figures" / "time_rrmse"
FULL_KEY = "wave_5000_501_256"

METHODS = {
    "AE": {
        "latent_key": "latent_AE_d30",
        "latent_model": WAVE_DIR / "models" / "AE_d30_wave_latent.pt",
        "decoder": "ae",
        "reduction_model": REDUCTION_DIR / "models" / "AE_d30_wave.pth",
    },
    "OAE": {
        "latent_key": "latent_DPN_d30",
        "latent_model": WAVE_DIR / "models" / "OAE_d30_wave_latent.pt",
        "decoder": "trunk",
        "trunk_key": "trunk_DPN_d30",
    },
    "FOAE": {
        "latent_key": "latent_DPN_d30_fno",
        "latent_model": WAVE_DIR / "models" / "FOAE_d30_wave_latent.pt",
        "decoder": "trunk",
        "trunk_key": "trunk_DPN_d30_fno",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute wave test-set RRMSE at each time index for AE, OAE, and FOAE "
            "latent prediction models, then save a CSV and a comparison plot."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--yscale", choices=["linear", "log"], default="linear")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-batch-size", type=int, default=200)
    parser.add_argument("--ymax-x1e3", type=float, default=7.5, help="Shared y-axis upper limit in units of 1e-3.")
    parser.add_argument("--font-size", type=float, default=20.0)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def patch_latent_fno_for_device() -> None:
    sys.path.insert(0, str(WAVE_DIR))
    from latent import FNO, SpectralConv2d_fast  # noqa: WPS433

    globals()["FNO"] = FNO
    globals()["SpectralConv2d_fast"] = SpectralConv2d_fast

    def get_grid(self, shape, device=None):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    def forward(self, x):
        grid = self.get_grid(x.shape, device=x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = torch.nn.functional.gelu(self.conv0(x) + self.w0(x))
        x = torch.nn.functional.gelu(self.conv1(x) + self.w1(x))
        x = torch.nn.functional.gelu(self.conv2(x) + self.w2(x))
        x = torch.nn.functional.gelu(self.conv3(x) + self.w3(x))
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x.squeeze(-1)

    FNO.get_grid = get_grid
    FNO.forward = forward


def import_ae_model_class() -> None:
    sys.path.insert(0, str(REDUCTION_DIR))
    import nn_ae_wave  # noqa: F401,WPS433

    # Some older checkpoints were saved when the AE module was imported as "nn".
    sys.modules.setdefault("nn", nn_ae_wave)
    globals()["AE"] = nn_ae_wave.AE
    globals()["ResidualBlock"] = nn_ae_wave.ResidualBlock


@torch.no_grad()
def predict_latent(model, x: torch.Tensor) -> torch.Tensor:
    return model(x.permute(0, 2, 1, 3)).permute(0, 2, 1)


@torch.no_grad()
def decode_latent(
    method: str,
    latent_prediction: torch.Tensor,
    handle: h5py.File,
    ae_model,
    device: torch.device,
) -> torch.Tensor:
    config = METHODS[method]
    if config["decoder"] == "trunk":
        trunk = torch.from_numpy(np.asarray(handle[config["trunk_key"]], dtype=np.float32)).to(device)
        return torch.einsum("bti,io->bto", latent_prediction, trunk)

    if ae_model is None:
        raise ValueError("AE decoder was not loaded.")
    batch_size, n_times, latent_dim = latent_prediction.shape
    flat = latent_prediction.reshape(batch_size * n_times, latent_dim)
    return ae_model.decode(flat).reshape(batch_size, n_times, -1)


def load_ae_decoder(device: torch.device):
    import_ae_model_class()
    model = torch_load(METHODS["AE"]["reduction_model"], device).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def compute_time_rrmse(args: argparse.Namespace, method: str, device: torch.device) -> dict[str, np.ndarray]:
    config = METHODS[method]
    latent_model = torch_load(config["latent_model"], device).to(device)
    latent_model.eval()
    for parameter in latent_model.parameters():
        parameter.requires_grad_(False)

    ae_model = load_ae_decoder(device) if config["decoder"] == "ae" else None

    with h5py.File(args.data_path, "r") as handle:
        full = handle[FULL_KEY]
        latent = handle[config["latent_key"]]
        n_cases, n_times, n_space = full.shape
        if args.ntest > n_cases:
            raise ValueError(f"ntest={args.ntest} is larger than available cases={n_cases}.")

        case_start = n_cases - args.ntest
        per_case_numerator = []
        per_case_denominator = []

        for start in range(case_start, n_cases, args.batch_size):
            stop = min(start + args.batch_size, n_cases)
            latent_first = torch.from_numpy(np.asarray(latent[start:stop, 0], dtype=np.float32)).to(device)
            target = torch.from_numpy(np.asarray(full[start:stop], dtype=np.float32)).to(device)
            x = latent_first.reshape(stop - start, 1, latent_first.shape[1], 1).repeat(1, n_times, 1, 1)

            latent_prediction = predict_latent(latent_model, x)
            prediction = decode_latent(method, latent_prediction, handle, ae_model, device)
            numerator = ((prediction - target) ** 2).sum(dim=2)
            denominator = (target**2).sum(dim=2)
            per_case_numerator.append(numerator.detach().cpu().numpy())
            per_case_denominator.append(denominator.detach().cpu().numpy())

    numerator = np.concatenate(per_case_numerator, axis=0).astype(np.float64)
    denominator = np.concatenate(per_case_denominator, axis=0).astype(np.float64)
    mean = np.sqrt(numerator.sum(axis=0) / denominator.sum(axis=0))
    lower, upper = bootstrap_aggregate_rrmse(
        numerator,
        denominator,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        batch_size=args.bootstrap_batch_size,
    )
    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
    }


def bootstrap_aggregate_rrmse(
    numerator: np.ndarray,
    denominator: np.ndarray,
    n_bootstrap: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_bootstrap <= 0:
        point = np.sqrt(numerator.sum(axis=0) / denominator.sum(axis=0))
        return point, point

    rng = np.random.default_rng(seed)
    n_cases = numerator.shape[0]
    draws = np.empty((n_bootstrap, numerator.shape[1]), dtype=np.float64)
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        indices = rng.integers(0, n_cases, size=(stop - start, n_cases))
        boot_num = numerator[indices].sum(axis=1)
        boot_den = denominator[indices].sum(axis=1)
        draws[start:stop] = np.sqrt(boot_num / boot_den)
    return np.percentile(draws, [2.5, 97.5], axis=0)


def write_csv(path: Path, curves: dict[str, dict[str, np.ndarray]]) -> None:
    methods = list(curves)
    n_times = len(next(iter(curves.values()))["mean"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["time_index"]
        for method in methods:
            header.extend([f"{method}_mean", f"{method}_lower_95", f"{method}_upper_95"])
        writer.writerow(header)
        for time_index in range(n_times):
            row = [time_index]
            for method in methods:
                row.extend(
                    [
                        f"{curves[method]['mean'][time_index]:.10e}",
                        f"{curves[method]['lower'][time_index]:.10e}",
                        f"{curves[method]['upper'][time_index]:.10e}",
                    ]
                )
            writer.writerow(row)


def shared_ylim(ymax_x1e3: float | None, curves: dict[str, dict[str, np.ndarray]]) -> tuple[float, float]:
    if ymax_x1e3 is not None and ymax_x1e3 > 0:
        return 0.0, ymax_x1e3 / 1e3
    max_upper = max(float(stats["upper"].max()) for stats in curves.values())
    upper_scaled = np.ceil(max_upper * 1e3 / 2.0) * 2.0
    return 0.0, upper_scaled / 1e3


def plot_method(
    path: Path,
    method: str,
    stats: dict[str, np.ndarray],
    yscale: str,
    dpi: int,
    ylim: tuple[float, float],
    font_size: float,
) -> None:
    time_index = np.arange(len(stats["mean"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    line = ax.plot(time_index, stats["mean"], linewidth=2.0)[0]
    ax.fill_between(
        time_index,
        stats["lower"],
        stats["upper"],
        color=line.get_color(),
        alpha=0.18,
        linewidth=0,
    )
    ax.set_xlabel("Time index")
    ax.set_ylabel("RRMSE")
    ax.set_yscale(yscale)
    ax.set_ylim(*ylim)
    if yscale == "linear":
        upper_scaled = ylim[1] * 1e3
        tick_step = 1.0
        yticks = np.arange(ylim[0] * 1e3, upper_scaled + tick_step * 0.5, tick_step) / 1e3
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value * 1e3:.0f}"))
        if method == "FOAE":
            ax.text(
                0.02,
                0.98,
                r"$\times 10^{-3}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=font_size,
            )
    ax.set_title(f"Wave test RRMSE by time - {method}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(axis="both", which="both", direction="in", labelsize=font_size * 0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    patch_latent_fno_for_device()
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": args.font_size,
            "axes.titlesize": args.font_size,
            "axes.labelsize": args.font_size,
            "xtick.labelsize": args.font_size * 0.9,
            "ytick.labelsize": args.font_size * 0.9,
            "mathtext.fontset": "stix",
        }
    )

    curves = {}
    for method in ["AE", "OAE", "FOAE"]:
        print(f"Computing {method} time-wise RRMSE on {device}...", flush=True)
        curves[method] = compute_time_rrmse(args, method, device)

    csv_path = args.output_dir / "wave_time_rrmse_ci.csv"
    write_csv(csv_path, curves)
    ylim = shared_ylim(args.ymax_x1e3, curves)
    for method, stats in curves.items():
        plot_method(
            args.output_dir / f"wave_time_rrmse_{method.lower()}_ci.png",
            method,
            stats,
            args.yscale,
            args.dpi,
            ylim,
            args.font_size,
        )
    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved plots: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
