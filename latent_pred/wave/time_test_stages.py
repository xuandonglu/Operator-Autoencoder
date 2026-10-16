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


THIS_FILE = Path(__file__).resolve()
WAVE_DIR = THIS_FILE.parent
PROJECT_ROOT = WAVE_DIR.parents[1]
REDUCTION_DIR = PROJECT_ROOT / "reduction"
DEFAULT_DATA_PATH = REDUCTION_DIR / "data" / "data_wave.h5df"
DEFAULT_OUTPUT_DIR = WAVE_DIR / "figures" / "test_timing"
FULL_KEY = "wave_5000_501_256"

METHODS = {
    "AE": {
        "latent_model": WAVE_DIR / "models" / "AE_d30_wave_latent.pt",
        "reduction_model": REDUCTION_DIR / "models" / "AE_d30_wave.pth",
        "reduction_module": "nn_ae_wave",
        "reconstruction": "ae_decoder",
    },
    "OAE": {
        "latent_model": WAVE_DIR / "models" / "OAE_d30_wave_latent.pt",
        "reduction_model": REDUCTION_DIR / "models" / "OAE_d30_wave.pth",
        "reduction_module": "nn_wave",
        "reconstruction": "trunk",
        "trunk_key": "trunk_DPN_d30",
    },
    "FOAE": {
        "latent_model": WAVE_DIR / "models" / "FOAE_d30_wave_latent.pt",
        "reduction_model": REDUCTION_DIR / "models" / "FOAE_d30_wave.pth",
        "reduction_module": "nn_wave",
        "reconstruction": "trunk",
        "trunk_key": "trunk_DPN_d30_fno",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Time wave test inference stages: full-field encoding, latent temporal model, "
            "and reconstruction from predicted latent trajectories."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=["AE", "OAE", "FOAE"])
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--start", type=int, default=None, help="Default uses the last ntest cases.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--decode-batch-size", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
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


def patch_reduction_module(module_name: str) -> None:
    sys.path.insert(0, str(REDUCTION_DIR))
    module = __import__(module_name)
    sys.modules["nn"] = module
    for name in ("AE", "DPN", "FNO1", "FNO2", "ResidualBlock", "SpectralConv1d"):
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
    model = torch_load(METHODS[method]["latent_model"], device).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def encode_in_batches(model, fields: torch.Tensor, batch_size: int) -> torch.Tensor:
    parts = []
    for start in range(0, fields.shape[0], batch_size):
        parts.append(model.encode(fields[start : start + batch_size]))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def predict_latent_in_batches(model, initial_latent: torch.Tensor, n_times: int, batch_size: int) -> torch.Tensor:
    parts = []
    for start in range(0, initial_latent.shape[0], batch_size):
        z0 = initial_latent[start : start + batch_size]
        x = z0.reshape(z0.shape[0], 1, z0.shape[1], 1).repeat(1, n_times, 1, 1)
        parts.append(model(x.permute(0, 2, 1, 3)).permute(0, 2, 1))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def reconstruct_ae_time_batched(ae_model, latent_trajectory: torch.Tensor, batch_size: int) -> torch.Tensor:
    parts = []
    for time_index in range(latent_trajectory.shape[1]):
        frame_latent = latent_trajectory[:, time_index]
        frame_parts = []
        for start in range(0, frame_latent.shape[0], batch_size):
            frame_parts.append(ae_model.decode(frame_latent[start : start + batch_size]))
        parts.append(torch.cat(frame_parts, dim=0).unsqueeze(1))
    return torch.cat(parts, dim=1)


@torch.no_grad()
def reconstruct_trunk_time_batched(latent_trajectory: torch.Tensor, trunk: torch.Tensor, batch_size: int) -> torch.Tensor:
    parts = []
    for time_index in range(latent_trajectory.shape[1]):
        frame_latent = latent_trajectory[:, time_index]
        frame_parts = []
        for start in range(0, frame_latent.shape[0], batch_size):
            frame_parts.append(torch.einsum("bi,io->bo", frame_latent[start : start + batch_size], trunk))
        parts.append(torch.cat(frame_parts, dim=0).unsqueeze(1))
    return torch.cat(parts, dim=1)


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
            stage_result = result[stage]
            rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "mean_s": stage_result["mean_s"],
                    "std_s": stage_result["std_s"],
                    "ntest": summary["ntest"],
                    "n_times": summary["n_times"],
                    "device": summary["device"],
                }
            )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    patch_latent_fno_for_device()
    decode_batch_size = args.batch_size if args.decode_batch_size is None else args.decode_batch_size

    with h5py.File(args.data_path, "r") as handle:
        n_cases, n_times, n_space = handle[FULL_KEY].shape
        start = n_cases - args.ntest if args.start is None else args.start
        stop = start + args.ntest
        if start < 0 or stop > n_cases:
            raise ValueError(f"Invalid case range [{start}, {stop}) for {n_cases} available cases.")
        fields = torch.from_numpy(np.asarray(handle[FULL_KEY][start:stop, 0], dtype=np.float32)).to(device)
        trunks = {
            method: torch.from_numpy(np.asarray(handle[config["trunk_key"]], dtype=np.float32)).to(device)
            for method, config in METHODS.items()
            if config["reconstruction"] == "trunk"
        }

    results = {}
    for method in args.methods:
        reduction_model = load_reduction_model(method, device)
        latent_model = load_latent_model(method, device)
        encode_times = []
        latent_times = []
        reconstruction_times = []
        total_times = []

        for repeat in range(args.warmup + args.repeats):
            encode_time, initial_latent = timed(
                device,
                lambda: encode_in_batches(reduction_model, fields, args.batch_size),
            )
            latent_time, latent_trajectory = timed(
                device,
                lambda z=initial_latent: predict_latent_in_batches(latent_model, z, n_times, args.batch_size),
            )
            if method == "AE":
                reconstruction_time, reconstruction = timed(
                    device,
                    lambda z=latent_trajectory: reconstruct_ae_time_batched(reduction_model, z, decode_batch_size),
                )
            else:
                reconstruction_time, reconstruction = timed(
                    device,
                    lambda z=latent_trajectory: reconstruct_trunk_time_batched(z, trunks[method], decode_batch_size),
                )

            total_time = encode_time + latent_time + reconstruction_time
            del initial_latent, latent_trajectory, reconstruction
            if device.type == "cuda":
                torch.cuda.empty_cache()

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

    summary = {
        "device": str(device),
        "data_path": str(args.data_path),
        "case_range": [start, stop],
        "ntest": args.ntest,
        "n_times": n_times,
        "n_space": n_space,
        "batch_size": args.batch_size,
        "decode_batch_size": decode_batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_scope": "HDF5 loading is excluded; tensors are already on the selected device before timing.",
        "stage_definition": {
            "encode": "u[:, 0, :] -> z0 using the reduction encoder; batch size equals ntest.",
            "latent_model": "z0 -> predicted latent trajectory z[:, :, :] using the latent FNO model.",
            "reconstruction": "all predicted time slices are reconstructed; each time slice is decoded with the same batch size as encode.",
        },
        "results": results,
    }
    json_path = args.output_dir / "wave_test_stage_timing.json"
    csv_path = args.output_dir / "wave_test_stage_timing.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(csv_path, summary)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
