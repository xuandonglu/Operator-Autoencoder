from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = THIS_DIR / "comp_frames_range_15.npz"
DEFAULT_OUTPUT = THIS_DIR / "data_cylinder_temp_xmask.h5df"

# Keep the legacy full-field key so the existing cylinder scripts can be pointed
# at this temporary file without changing their FULL_KEY constants.
FULL_KEY = "uxuy_50_300_1_768_512"
MASK_KEY = "mask_768_512"
SOURCE_KEY = "arr_0"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cylinder x-velocity and mask from comp_frames_range_15.npz."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_or_replace(handle: h5py.File, key: str, value: np.ndarray) -> None:
    if key in handle:
        if handle[key].shape == value.shape:
            handle[key][...] = value
            return
        del handle[key]
    handle.create_dataset(key, data=value, dtype=value.dtype, compression="gzip", compression_opts=4)


def main() -> None:
    args = parse_args()
    source = np.load(args.input)[SOURCE_KEY]
    if source.ndim != 3 or source.shape[1:] != (IMAGE_HEIGHT * IMAGE_WIDTH, 4):
        raise ValueError(f"Unexpected input shape {source.shape}; expected [n_frames, 393216, 4].")

    frames = source.reshape(source.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 4)
    ux = frames[..., 0].transpose(0, 2, 1).astype(np.float32)
    mask = frames[0, ..., 3].T.astype(np.float32)
    if not np.all(frames[..., 3] == frames[0, ..., 3]):
        raise ValueError("Mask is not constant across frames.")

    full = ux.reshape(1, ux.shape[0], 1, IMAGE_WIDTH, IMAGE_HEIGHT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        write_or_replace(handle, FULL_KEY, full)
        write_or_replace(handle, MASK_KEY, mask)
        handle.attrs["source_npz"] = str(args.input)
        handle.attrs["source_key"] = SOURCE_KEY
        handle.attrs["layout"] = "[case, time, channel, x, y]"
        handle.attrs["note"] = "Temporary cylinder data made from x-velocity only; mask is stored separately."

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "full_key": FULL_KEY,
        "full_shape": list(full.shape),
        "full_dtype": str(full.dtype),
        "mask_key": MASK_KEY,
        "mask_shape": list(mask.shape),
        "mask_unique": np.unique(mask).astype(float).tolist(),
        "ux_min": float(ux.min()),
        "ux_max": float(ux.max()),
        "ux_mean": float(ux.mean()),
        "ux_std": float(ux.std()),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
