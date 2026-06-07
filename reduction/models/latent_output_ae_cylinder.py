from pathlib import Path
import sys

import h5py
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REDUCTION_DIR = THIS_DIR.parent
sys.path.insert(0, str(REDUCTION_DIR))
import nn_ae_cylinder  # noqa: F401 - required when loading the pickled model


MODEL_PATH = THIS_DIR / "AE_d140_cylinder.pth"
DATA_PATH = REDUCTION_DIR / "data" / "data_cylinder.h5df"
FULL_KEY = "uxuy_50_300_1_768_512"
LATENT_KEY = "latent_AE_d140_cylinder"


def write_or_replace(handle, key, value):
    if key in handle:
        if handle[key].shape == value.shape:
            handle[key][...] = value
            return
        del handle[key]
    handle.create_dataset(key, data=value, dtype="float32")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH, weights_only=False, map_location=device).to(device)
    model.eval()
    latent_dim = int(model.latent_dim)

    with h5py.File(DATA_PATH, "a") as handle:
        full = handle[FULL_KEY]
        n_case, n_time = full.shape[:2]
        latent = np.empty((n_case, n_time, latent_dim), dtype=np.float32)
        with torch.no_grad():
            for case_id in range(n_case):
                for start in range(0, n_time, 1):
                    batch = torch.from_numpy(np.asarray(full[case_id, start:start + 1], dtype=np.float32)).to(device)
                    z = model.encode(batch).reshape(1, latent_dim)
                    latent[case_id, start:start + 1] = z.detach().cpu().numpy().astype(np.float32)
                print(f"Sample {case_id}")
        write_or_replace(handle, LATENT_KEY, latent)


if __name__ == "__main__":
    main()
