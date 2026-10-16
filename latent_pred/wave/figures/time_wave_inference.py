from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch

from generate_wave_sample_folders import (
    OAE_ROOT,
    AE_DATA,
    AE_LATENT_KEY,
    AE_REDUCTION_DIR,
    AE_REDUCTION_MODEL,
    AE_TEMPORAL,
    DM_DATA,
    F_OAE_TEMPORAL,
    F_OAE_TRUNK_KEY,
    FNO,
    FULL_KEY,
    OAE_TEMPORAL,
    OAE_TRUNK_KEY,
    SpectralConv2d_fast,
    decode_dpn,
    load_temporal_model,
    predict_latent,
)


THIS_FILE = Path(__file__).resolve()
WORKSPACE = OAE_ROOT.parent
DM_REDUCTION_DIR = OAE_ROOT / "reduction"
DM_REDUCTION_MODELS = OAE_ROOT / "reduction" / "models"
OUT_PATH = THIS_FILE.parent / "wave_inference_timing_1000.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time wave encode, latent prediction, and reconstruction stages.")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--start", type=int, default=4000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--ae-decode-batch", type=int, default=65536)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn) -> tuple[float, object]:
    sync(device)
    start = perf_counter()
    out = fn()
    sync(device)
    return perf_counter() - start, out


def load_pickled_model_with_nn(model_path: Path, nn_dir: Path, device: torch.device):
    sys.path.insert(0, str(nn_dir))
    try:
        from checkpoint_utils import load_reduction_checkpoint
        method = 'AE' if model_path.name.startswith('AE_') else 'OAE'
        model = load_reduction_checkpoint(model_path, 'wave', method, device)
        model.eval()
        return model
    finally:
        if str(nn_dir) in sys.path:
            sys.path.remove(str(nn_dir))


def delete_nn_module() -> None:
    if "nn" in sys.modules:
        del sys.modules["nn"]


def load_dpn_reduction_model(name: str, device: torch.device):
    delete_nn_module()
    return load_pickled_model_with_nn(DM_REDUCTION_MODELS / name, DM_REDUCTION_DIR, device)


def load_ae_reduction_model(device: torch.device):
    delete_nn_module()
    return load_pickled_model_with_nn(AE_REDUCTION_MODEL, AE_REDUCTION_DIR, device)


@torch.no_grad()
def encode_in_batches(model, inputs: torch.Tensor, batch_size: int) -> torch.Tensor:
    parts = []
    for start in range(0, inputs.shape[0], batch_size):
        parts.append(model.encode(inputs[start : start + batch_size]))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def infer_in_batches(model, initial_latent: torch.Tensor, n_times: int, batch_size: int) -> torch.Tensor:
    parts = []
    for start in range(0, initial_latent.shape[0], batch_size):
        parts.append(predict_latent(model, initial_latent[start : start + batch_size], n_times))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def decode_dpn_in_batches(latent: torch.Tensor, trunk: torch.Tensor, batch_size: int) -> torch.Tensor:
    parts = []
    for start in range(0, latent.shape[0], batch_size):
        parts.append(decode_dpn(latent[start : start + batch_size], trunk))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def decode_ae_in_chunks(latent: torch.Tensor, ae_model, chunk_size: int) -> torch.Tensor:
    flat = latent.reshape(-1, latent.shape[-1])
    parts = []
    for start in range(0, flat.shape[0], chunk_size):
        parts.append(ae_model.decode(flat[start : start + chunk_size]))
    return torch.cat(parts, dim=0).reshape(latent.shape[0], latent.shape[1], -1)


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.set_num_threads(4)

    with h5py.File(DM_DATA, "r") as dm, h5py.File(AE_DATA, "r") as ae_data:
        fields = torch.from_numpy(
            np.asarray(dm[FULL_KEY][args.start : args.start + args.n_samples, 0], dtype=np.float32)
        ).to(device)
        # Stored initial latents are used only for an optional inference-only timing reference.
        stored_initial = {
            "foae": torch.from_numpy(np.asarray(dm["latent_DPN_d30_fno"][args.start : args.start + args.n_samples, 0], dtype=np.float32)).to(device),
            "oae": torch.from_numpy(np.asarray(dm["latent_DPN_d30"][args.start : args.start + args.n_samples, 0], dtype=np.float32)).to(device),
            "ae": torch.from_numpy(np.asarray(ae_data[AE_LATENT_KEY][args.start : args.start + args.n_samples, 0], dtype=np.float32)).to(device),
        }
        trunks = {
            "foae": torch.from_numpy(np.asarray(dm[F_OAE_TRUNK_KEY][:], dtype=np.float32)).to(device),
            "oae": torch.from_numpy(np.asarray(dm[OAE_TRUNK_KEY][:], dtype=np.float32)).to(device),
        }
        n_times = dm[FULL_KEY].shape[1]

    reduction = {
        "foae": load_dpn_reduction_model("FOAE_d30_wave.pth", device),
        "oae": load_dpn_reduction_model("OAE_d30_wave.pth", device),
        "ae": load_ae_reduction_model(device),
    }
    temporal = {
        "foae": load_temporal_model(F_OAE_TEMPORAL, device),
        "oae": load_temporal_model(OAE_TEMPORAL, device),
        "ae": load_temporal_model(AE_TEMPORAL, device),
    }

    results: dict[str, dict[str, object]] = {}
    for method in ("foae", "oae", "ae"):
        encode_times: list[float] = []
        infer_times: list[float] = []
        reconstruct_times: list[float] = []

        for repeat in range(args.warmup + args.repeats):
            encode_time, initial_latent = timed(
                device,
                lambda method=method: encode_in_batches(reduction[method], fields, args.batch_size),
            )
            infer_time, trajectory_latent = timed(
                device,
                lambda method=method, initial_latent=initial_latent: infer_in_batches(
                    temporal[method], initial_latent, n_times, args.batch_size
                ),
            )
            if method == "ae":
                reconstruct_time, reconstructed = timed(
                    device,
                    lambda trajectory_latent=trajectory_latent: decode_ae_in_chunks(
                        trajectory_latent, reduction["ae"], args.ae_decode_batch
                    ),
                )
            else:
                reconstruct_time, reconstructed = timed(
                    device,
                    lambda method=method, trajectory_latent=trajectory_latent: decode_dpn_in_batches(
                        trajectory_latent, trunks[method], args.batch_size
                    ),
                )
            # Keep the tensors live until the timed section finishes; then free memory.
            del initial_latent, trajectory_latent, reconstructed
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if repeat >= args.warmup:
                encode_times.append(encode_time)
                infer_times.append(infer_time)
                reconstruct_times.append(reconstruct_time)
                print(
                    json.dumps(
                        {
                            "method": method,
                            "repeat": repeat - args.warmup,
                            "encoding_s": encode_time,
                            "latent_inference_s": infer_time,
                            "reconstruction_s": reconstruct_time,
                        }
                    ),
                    flush=True,
                )

        encode_mean, encode_std = mean_std(encode_times)
        infer_mean, infer_std = mean_std(infer_times)
        reconstruct_mean, reconstruct_std = mean_std(reconstruct_times)
        results[method] = {
            "encoding_s": {"mean": encode_mean, "std": encode_std, "runs": encode_times},
            "latent_inference_s": {"mean": infer_mean, "std": infer_std, "runs": infer_times},
            "reconstruction_s": {"mean": reconstruct_mean, "std": reconstruct_std, "runs": reconstruct_times},
            "encoded_inputs": args.n_samples,
            "predicted_trajectories": args.n_samples,
            "predicted_frames": args.n_samples * n_times,
            "throughput": {
                "encoded_inputs_per_s": args.n_samples / encode_mean,
                "predicted_trajectories_per_s": args.n_samples / infer_mean,
                "reconstructed_frames_per_s": (args.n_samples * n_times) / reconstruct_mean,
            },
        }

        # A small guard against the optimizer eliminating stored_initial from consideration in future edits.
        _ = stored_initial[method].shape

    summary = {
        "device": str(device),
        "n_samples": args.n_samples,
        "n_times": n_times,
        "predicted_frames": args.n_samples * n_times,
        "start_index": args.start,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "batch_size": args.batch_size,
        "ae_decode_batch": args.ae_decode_batch,
        "timing_scope": "GPU model stages only; HDF5 loading is excluded.",
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT_PATH), "results": results}), flush=True)


if __name__ == "__main__":
    main()
