from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from PIL import Image
import torch

from generate_kdv_7row_panels import (
    AE_DATA,
    DM_DATA,
    OUT_DIR,
    FNO,
    SpectralConv2d_fast,
    load_ae_latent_model,
    load_ae_reduction_model,
    load_dm_latent_model,
    load_dpn_reduction_model,
    mse,
    patch_cuda_for_cpu,
    predict_ae_latent,
    predict_dpn_latent,
    reconstruct_ae,
    reconstruct_dpn,
)


CMAP_NAME = "RdBu"  # low values red, high values blue, white near the middle.
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 1024  # x length is twice the t length.


def load_selected_samples() -> list[int]:
    manifest = OUT_DIR / "manifest.json"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as f:
            return [int(v) for v in json.load(f)["selected_sample_indices_zero_based"]]
    return [4902, 4905, 4936, 4970, 4993]


def choose_scale(values: list[float]) -> int:
    max_abs = max(abs(v) for v in values)
    if max_abs == 0:
        return 0
    exponent = int(math.floor(math.log10(max_abs)))
    if 1 <= max_abs < 100:
        return 0
    return exponent


def rounded_limits(vmin: float, vmax: float, force_zero_bottom: bool = False) -> tuple[float, float, float, int, tuple[float, float, float]]:
    exponent = choose_scale([vmin, vmax])
    scale = 10.0**exponent
    svmin = vmin / scale
    svmax = vmax / scale
    low_i = 0 if force_zero_bottom else math.floor(svmin * 10.0)
    high_i = math.ceil(svmax * 10.0)
    if high_i == low_i:
        high_i += 1

    # Make the middle tick land exactly on one decimal place.
    if (low_i + high_i) % 2 != 0:
        if abs(high_i) <= abs(low_i):
            high_i += 1
        elif not force_zero_bottom:
            low_i -= 1
        else:
            high_i += 1

    low = low_i / 10.0 * scale
    high = high_i / 10.0 * scale
    mid = 0.5 * (low + high)
    labels = (high / scale, mid / scale, low / scale)
    return low, high, mid, exponent, labels


def save_field_image(array: np.ndarray, path: Path, vmin: float, vmax: float) -> None:
    cmap = plt.get_cmap(CMAP_NAME)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(array), bytes=True)
    img = Image.fromarray(rgba).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BICUBIC)
    img.save(path)


def save_colorbar(path: Path, vmin: float, vmax: float, mid: float, exponent: int, labels: tuple[float, float, float]) -> None:
    fig = plt.figure(figsize=(2.75, 1.12), dpi=300)
    cax = fig.add_axes([0.12, 0.72, 0.76, 0.15])
    sm = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=CMAP_NAME)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([vmin, mid, vmax])
    cb.set_ticklabels([f"{labels[2]:.1f}", f"{labels[1]:.1f}", f"{labels[0]:.1f}"])
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.tick_params(labelsize=12, length=0, pad=3)
    cb.outline.set_linewidth(0.8)
    if exponent != 0:
        fig.text(0.64, 0.62, rf"$\times 10^{{{exponent}}}$", fontsize=10, ha="left", va="center")
    fig.savefig(path, transparent=False)
    plt.close(fig)


def clean_sample_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("*.png"):
        old.unlink()


def save_sample_outputs(
    sample_dir: Path,
    benchmark: np.ndarray,
    foae: np.ndarray,
    oae: np.ndarray,
    ae: np.ndarray,
) -> dict[str, float]:
    clean_sample_dir(sample_dir)
    err_foae = np.abs(foae - benchmark)
    err_oae = np.abs(oae - benchmark)
    err_ae = np.abs(ae - benchmark)
    fields = [benchmark, foae, oae, ae]
    errors = [err_foae, err_oae, err_ae]
    field_vmin, field_vmax, field_mid, field_exp, field_labels = rounded_limits(
        float(min(x.min() for x in fields)),
        float(max(x.max() for x in fields)),
    )
    error_vmin, error_vmax, error_mid, error_exp, error_labels = rounded_limits(
        0.0,
        float(max(x.max() for x in errors)),
        force_zero_bottom=True,
    )

    save_field_image(benchmark, sample_dir / "01_benchmark.png", field_vmin, field_vmax)
    save_field_image(foae, sample_dir / "02_F-OAE.png", field_vmin, field_vmax)
    save_field_image(oae, sample_dir / "03_OAE.png", field_vmin, field_vmax)
    save_field_image(ae, sample_dir / "04_AE.png", field_vmin, field_vmax)

    mse_foae = mse(foae, benchmark)
    mse_oae = mse(oae, benchmark)
    mse_ae = mse(ae, benchmark)
    save_field_image(err_foae, sample_dir / f"05_F-OAE_error_MSE_{mse_foae:.3e}.png", error_vmin, error_vmax)
    save_field_image(err_oae, sample_dir / f"06_OAE_error_MSE_{mse_oae:.3e}.png", error_vmin, error_vmax)
    save_field_image(err_ae, sample_dir / f"07_AE_error_MSE_{mse_ae:.3e}.png", error_vmin, error_vmax)

    save_colorbar(
        sample_dir / "field_colorbar.png",
        field_vmin,
        field_vmax,
        field_mid,
        field_exp,
        field_labels,
    )
    save_colorbar(
        sample_dir / "error_colorbar.png",
        error_vmin,
        error_vmax,
        error_mid,
        error_exp,
        error_labels,
    )

    return {
        "mse_foae": mse_foae,
        "mse_oae": mse_oae,
        "mse_ae": mse_ae,
        "field_vmin": field_vmin,
        "field_vmax": field_vmax,
        "field_mid": field_mid,
        "field_exponent": field_exp,
        "error_vmin": error_vmin,
        "error_vmax": error_vmax,
        "error_mid": error_mid,
        "error_exponent": error_exp,
    }


def main() -> None:
    patch_cuda_for_cpu()
    torch.set_num_threads(4)
    selected = load_selected_samples()

    dpn_fno = load_dpn_reduction_model("FOAE_d50_kdv.pth")
    dpn_plain = load_dpn_reduction_model("OAE_d50_kdv.pth")
    ae = load_ae_reduction_model("AE_d50_kdv.pth")

    latent_fno = load_dm_latent_model("FOAE_d50_kdv_latent.pt")
    latent_plain = load_dm_latent_model("OAE_d50_kdv_latent.pt")
    latent_ae = load_ae_latent_model("AE_d50_kdv_latent.pt")

    summary = {
        "selected_sample_indices_zero_based": selected,
        "image_size_pixels": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "cmap": CMAP_NAME,
        "colorbar": "two horizontal scales per folder: one shared by response fields and one shared by absolute-error maps",
        "reconstruction": {},
        "latent_prediction": {},
    }

    with h5py.File(DM_DATA, "r") as f_dm, h5py.File(AE_DATA, "r") as f_ae:
        clean = f_dm["kdv_5000_501_512"]
        noisy = f_dm["kdv_5000_501_512"]
        latent_dpn_fno = f_dm["latent_DPN_d50_fno"]
        latent_dpn_plain = f_dm["latent_DPN_d50"]
        trunk_dpn_fno = f_dm["trunk_DPN_d50_fno"][:]
        trunk_dpn_plain = f_dm["trunk_DPN_d50"][:]
        latent_ae_data = f_ae["latent_fno_d50_m20_w1"]

        for idx in selected:
            benchmark = clean[idx].astype(np.float32)
            noisy_sample = noisy[idx].astype(np.float32)

            rec_f = reconstruct_dpn(dpn_fno, noisy_sample)
            rec_o = reconstruct_dpn(dpn_plain, noisy_sample)
            rec_a = reconstruct_ae(ae, noisy_sample)
            rec_dir = OUT_DIR / f"sample_{idx:04d}" / "reconstruction"
            summary["reconstruction"][str(idx)] = save_sample_outputs(rec_dir, benchmark, rec_f, rec_o, rec_a)

            pred_f = predict_dpn_latent(latent_fno, latent_dpn_fno[idx, 0], trunk_dpn_fno, benchmark.shape[0])
            pred_o = predict_dpn_latent(latent_plain, latent_dpn_plain[idx, 0], trunk_dpn_plain, benchmark.shape[0])
            pred_a = predict_ae_latent(latent_ae, ae, latent_ae_data[idx, 0], benchmark.shape[0])
            pred_dir = OUT_DIR / f"sample_{idx:04d}" / "latent_prediction"
            summary["latent_prediction"][str(idx)] = save_sample_outputs(pred_dir, benchmark, pred_f, pred_o, pred_a)

            print(json.dumps({"sample": idx, "done": True}), flush=True)

    with (OUT_DIR / "sample_folders_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"output": str(OUT_DIR), "samples": selected}))


if __name__ == "__main__":
    main()
