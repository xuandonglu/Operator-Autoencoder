from __future__ import annotations

import json
import sys

import h5py
import numpy as np
import torch

from generate_kdv_7row_panels import (
    AE_DATA,
    DM_DATA,
    FNO,
    SpectralConv2d_fast,
    load_ae_latent_model,
    load_ae_reduction_model,
    load_dm_latent_model,
    patch_cuda_for_cpu,
    predict_ae_latent,
    predict_dpn_latent,
)


def main() -> None:
    patch_cuda_for_cpu()
    sys.modules["__main__"].FNO = FNO
    sys.modules["__main__"].SpectralConv2d_fast = SpectralConv2d_fast
    torch.set_num_threads(4)
    ae = load_ae_reduction_model("fno_d50_m20_w1.pth")
    latent_foae = load_dm_latent_model("kdv1d_N4900_p11_m26_1e-1pi_DPN_d50_fno")
    latent_oae = load_dm_latent_model("kdv1d_N4900_p11_m26_1e-3pi_DPN_d50")
    latent_ae = load_ae_latent_model("kdv1d_sr_N4900_p11_m26_1e1pi_fno_d50_m20_w1")
    sums = {"foae": 0.0, "oae": 0.0, "ae": 0.0}
    count = 0
    with h5py.File(DM_DATA, "r") as f_dm, h5py.File(AE_DATA, "r") as f_ae:
        clean = f_dm["kdv_5000_501_512"]
        latent_dpn_fno = f_dm["latent_DPN_d50_fno"]
        latent_dpn_plain = f_dm["latent_DPN_d50"]
        trunk_dpn_fno = f_dm["trunk_DPN_d50_fno"][:]
        trunk_dpn_plain = f_dm["trunk_DPN_d50"][:]
        latent_ae_data = f_ae["latent_fno_d50_m20_w1"]
        for idx in range(4900, 5000):
            benchmark = clean[idx].astype(np.float32)
            predictions = {
                "foae": predict_dpn_latent(
                    latent_foae, latent_dpn_fno[idx, 0], trunk_dpn_fno, benchmark.shape[0]
                ),
                "oae": predict_dpn_latent(
                    latent_oae, latent_dpn_plain[idx, 0], trunk_dpn_plain, benchmark.shape[0]
                ),
                "ae": predict_ae_latent(latent_ae, ae, latent_ae_data[idx, 0], benchmark.shape[0]),
            }
            for name, prediction in predictions.items():
                sums[name] += float(np.square(prediction - benchmark).sum(dtype=np.float64))
            count += benchmark.size
            if (idx + 1) % 10 == 0:
                print(json.dumps({"processed_through_zero_based_sample": idx}), flush=True)
    summary = {
        "test_sample_indices_zero_based": [4900, 4999],
        "test_sample_count": 100,
        "field_shape_per_sample": [501, 512],
        "overall_mse": {name: value / count for name, value in sums.items()},
    }
    output = __import__("pathlib").Path(__file__).with_name("kdv_100_prediction_mse.json")
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
