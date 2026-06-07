from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
GENERATOR = THIS_FILE.parent / "generate_sample_folders.py"
OUT_PATH = THIS_FILE.parent / "sample_folders" / "heldout_case_mse.json"


def import_generator():
    spec = importlib.util.spec_from_file_location("cylinder_hist10_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def decode_all_steps(module, method, predicted_z, trunks, ae_decoder, spatial_shape):
    flat = predicted_z.reshape(-1, predicted_z.shape[-1])
    if method == "ae":
        return ae_decoder.decode(flat).reshape(predicted_z.shape[0], predicted_z.shape[1], *spatial_shape)
    return module.decode_dpn_batch(flat, trunks[method]).reshape(
        predicted_z.shape[0], predicted_z.shape[1], *spatial_shape
    )


@torch.no_grad()
def main() -> None:
    module = import_generator()
    device = module.resolve_device()
    predictors = {method: module.load_temporal_model(method, device) for method in module.METHODS}
    ae_decoder = module.load_ae_decoder(device)
    rows = []

    with h5py.File(module.DATA_PATH, "r") as handle:
        full = handle[module.FULL_KEY]
        spatial_shape = (full.shape[-2], full.shape[-1])
        stop = full.shape[1] - module.HISTORY - module.FUTURE_STEPS + 1
        starts = np.arange(stop)
        latents = {
            method: np.asarray(handle[info["latent"]][:], dtype=np.float32)
            for method, info in module.METHODS.items()
        }
        trunks = {
            method: module.load_trunks(handle, str(info["trunk"]), device)
            for method, info in module.METHODS.items()
            if "trunk" in info
        }

        for case in range(40, 50):
            row = {"case_zero_based": case}
            for method in ("foae", "oae", "ae"):
                batch_size = 2 if method == "ae" else 8
                all_sse = 0.0
                all_count = 0
                last_sse = 0.0
                last_count = 0
                for offset in range(0, stop, batch_size):
                    batch_starts = starts[offset : offset + batch_size]
                    windows = np.stack(
                        [latents[method][case, start : start + module.HISTORY] for start in batch_starts],
                        axis=0,
                    )
                    predicted_z = module.predict_latent(predictors[method], windows, device)
                    predicted = decode_all_steps(module, method, predicted_z, trunks, ae_decoder, spatial_shape)
                    target = np.stack(
                        [
                            np.asarray(
                                full[case, start + module.HISTORY : start + module.HISTORY + module.FUTURE_STEPS, 0],
                                dtype=np.float32,
                            )
                            for start in batch_starts
                        ],
                        axis=0,
                    )
                    truth = torch.from_numpy(target).to(device)
                    diff = predicted - truth
                    all_sse += float(torch.square(diff).sum().cpu())
                    all_count += diff.numel()
                    last = diff[:, -1]
                    last_sse += float(torch.square(last).sum().cpu())
                    last_count += last.numel()
                row[f"{method}_all10_field_mse"] = all_sse / max(1, all_count)
                row[f"{method}_last_step_field_mse"] = last_sse / max(1, last_count)
            rows.append(row)
            print(json.dumps(row), flush=True)

    summary = {
        "meaning": "full-field MSE on cylinder held-out experiments for truth-input 10->10 latent prediction",
        "cases_zero_based": [40, 49],
        "all10_field_mse": {
            method: float(np.mean([row[f"{method}_all10_field_mse"] for row in rows]))
            for method in ("foae", "oae", "ae")
        },
        "last_step_field_mse": {
            method: float(np.mean([row[f"{method}_last_step_field_mse"] for row in rows]))
            for method in ("foae", "oae", "ae")
        },
        "per_case": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT_PATH), "summary": summary["all10_field_mse"]}), flush=True)


if __name__ == "__main__":
    main()
