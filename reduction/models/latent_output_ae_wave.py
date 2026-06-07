from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
MODELS_DIR = THIS_FILE.parent
REDUCTION_DIR = MODELS_DIR.parent
DEFAULT_DATA_PATH = REDUCTION_DIR / "data" / "data.hdf5"
DEFAULT_MODEL_PATH = MODELS_DIR / "fno_d30.pth"
FULL_KEY = "wave_5000_501_256"
LATENT_KEY = "latent_AE_d30"

sys.path.insert(0, str(REDUCTION_DIR))
from nn import AE  # noqa: E402,F401 - needed when loading the pickled model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode the wave data with the trained AE d30 model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--case-batch-size", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = torch.load(args.model_path, map_location=device, weights_only=False).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with h5py.File(args.data_path, "a") as handle:
        source = handle[FULL_KEY]
        n_cases, n_times, _ = source.shape
        shape = (n_cases, n_times, int(model.latent_dim))
        if LATENT_KEY in handle:
            if handle[LATENT_KEY].shape != shape:
                del handle[LATENT_KEY]
                latent = handle.create_dataset(LATENT_KEY, shape=shape, dtype="float32")
            else:
                latent = handle[LATENT_KEY]
        else:
            latent = handle.create_dataset(LATENT_KEY, shape=shape, dtype="float32")

        for case_start in range(0, n_cases, args.case_batch_size):
            case_end = min(case_start + args.case_batch_size, n_cases)
            block = np.asarray(source[case_start:case_end], dtype=np.float32)
            flat = torch.from_numpy(block.reshape(-1, block.shape[-1]))
            encoded = []
            for batch_start in range(0, flat.shape[0], args.encode_batch_size):
                batch = flat[batch_start : batch_start + args.encode_batch_size].to(device, non_blocking=True)
                encoded.append(model.encode(batch).cpu().numpy().astype(np.float32))
            latent[case_start:case_end] = np.concatenate(encoded, axis=0).reshape(case_end - case_start, n_times, -1)
            print({"encoded_cases": [case_start, case_end - 1]}, flush=True)

    print(
        {
            "data_path": str(args.data_path),
            "model_path": str(args.model_path),
            "source": FULL_KEY,
            "dataset": LATENT_KEY,
            "shape": shape,
            "device": str(device),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
