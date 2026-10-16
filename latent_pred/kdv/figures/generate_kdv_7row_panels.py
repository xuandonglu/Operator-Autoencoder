from __future__ import annotations

import json
import sys
from pathlib import Path
import importlib

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


THIS_FILE = Path(__file__).resolve()
OAE_ROOT = THIS_FILE.parents[3]
REPO_ROOT = OAE_ROOT.parent
OUT_DIR = THIS_FILE.parent
RECON_DIR = OUT_DIR / "reconstruction"
LATENT_DIR = OUT_DIR / "latent_prediction"

DM_ROOT = OAE_ROOT
AE_ROOT = OAE_ROOT

DM_DATA = OAE_ROOT / "reduction" / "data" / "data_kdv.h5df"
AE_DATA = DM_DATA

DM_REDUCTION_MODELS = OAE_ROOT / "reduction" / "models"
AE_REDUCTION_MODELS = OAE_ROOT / "reduction" / "models"
DM_LATENT_MODELS = OAE_ROOT / "latent_pred" / "kdv" / "models"
AE_LATENT_MODELS = OAE_ROOT / "latent_pred" / "kdv" / "models"


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.conv0 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc0 = nn.Linear(3, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape).to(x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.squeeze(-1)

    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float32)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)


def safe_torch_load(path: Path):
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def patch_cuda_for_cpu() -> None:
    if torch.cuda.is_available():
        return
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self


def prepend_path(path: Path) -> None:
    text = str(path)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


def load_dpn_reduction_model(filename: str):
    prepend_path(DM_ROOT / "reduction")
    from checkpoint_utils import load_reduction_checkpoint
    model = load_reduction_checkpoint(DM_REDUCTION_MODELS / filename, 'kdv', 'OAE', 'cpu')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_ae_reduction_model(filename: str):
    prepend_path(AE_ROOT / "reduction")
    from checkpoint_utils import load_reduction_checkpoint
    model = load_reduction_checkpoint(AE_REDUCTION_MODELS / filename, 'kdv', 'AE', 'cpu')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_dm_latent_model(filename: str):
    prepend_path(DM_ROOT / "latent_pred" / "kdv")
    model = safe_torch_load(DM_LATENT_MODELS / filename)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_ae_latent_model(filename: str):
    prepend_path(AE_ROOT / "latent_pred" / "kdv")
    model = safe_torch_load(AE_LATENT_MODELS / filename)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def reconstruct_dpn(model, sample: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(sample.astype(np.float32))
    out = model(x).detach().cpu().numpy()
    return out.astype(np.float32)


@torch.no_grad()
def reconstruct_ae(model, sample: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(sample.astype(np.float32)).unsqueeze(1)
    out = model(x).detach().cpu()
    if out.ndim == 3:
        out = out.squeeze(1)
    return out.numpy().astype(np.float32)


@torch.no_grad()
def predict_dpn_latent(model, latent0: np.ndarray, trunk: np.ndarray, t_steps: int) -> np.ndarray:
    z0 = torch.from_numpy(latent0.astype(np.float32))
    x = z0.reshape(1, 1, z0.shape[0], 1).repeat(1, t_steps, 1, 1)
    out_z = model(x.permute(0, 2, 1, 3)).permute(0, 2, 1).detach().cpu()
    tr = torch.from_numpy(trunk.astype(np.float32))
    out = torch.einsum("bti,io->bto", out_z, tr)
    return out.squeeze(0).numpy().astype(np.float32)


@torch.no_grad()
def predict_ae_latent(model, ae_model, latent0: np.ndarray, t_steps: int) -> np.ndarray:
    z0 = torch.from_numpy(latent0.astype(np.float32))
    x = z0.reshape(1, 1, z0.shape[0], 1).repeat(1, t_steps, 1, 1)
    out_z = model(x.permute(0, 2, 1, 3)).permute(0, 2, 1).detach().cpu()
    z = out_z.reshape(t_steps, out_z.shape[-1])
    out = ae_model.decode(z).detach().cpu()
    if out.ndim == 3:
        out = out.squeeze(1)
    return out.numpy().astype(np.float32)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def plot_panel(path: Path, benchmark: np.ndarray, foae: np.ndarray, oae: np.ndarray, ae: np.ndarray, title: str) -> dict:
    rows = [
        ("benchmark", benchmark),
        ("F-OAE", foae),
        ("OAE", oae),
        ("AE", ae),
        ("F-OAE err", np.abs(foae - benchmark)),
        ("OAE err", np.abs(oae - benchmark)),
        ("AE err", np.abs(ae - benchmark)),
    ]
    field_stack = np.stack([benchmark, foae, oae, ae])
    field_vmin = float(np.nanmin(field_stack))
    field_vmax = float(np.nanmax(field_stack))
    err_stack = np.stack([np.abs(foae - benchmark), np.abs(oae - benchmark), np.abs(ae - benchmark)])
    err_vmin = 0.0
    err_vmax = float(np.nanmax(err_stack))
    metrics = {
        "mse_foae": mse(foae, benchmark),
        "mse_oae": mse(oae, benchmark),
        "mse_ae": mse(ae, benchmark),
        "field_vmin": field_vmin,
        "field_vmax": field_vmax,
        "error_vmax": err_vmax,
    }

    fig, axes = plt.subplots(
        nrows=7,
        ncols=1,
        figsize=(11.5, 10.2),
        constrained_layout=False,
        sharex=True,
        sharey=True,
    )
    for i, (label, arr) in enumerate(rows):
        if i < 4:
            im = axes[i].imshow(arr, aspect="auto", origin="lower", cmap="RdBu_r", vmin=field_vmin, vmax=field_vmax)
        else:
            im = axes[i].imshow(arr, aspect="auto", origin="lower", cmap="magma", vmin=err_vmin, vmax=err_vmax)
        axes[i].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10)
        axes[i].tick_params(labelsize=8, length=2)
    axes[-1].set_xlabel("x index")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    fig.supylabel("time index", fontsize=11)
    fig.suptitle(
        f"{title} | MSE F-OAE={metrics['mse_foae']:.3e}, OAE={metrics['mse_oae']:.3e}, AE={metrics['mse_ae']:.3e}",
        fontsize=12,
    )
    cax1 = fig.add_axes([0.915, 0.548, 0.016, 0.330])
    cax2 = fig.add_axes([0.915, 0.122, 0.016, 0.245])
    fig.colorbar(axes[0].images[0], cax=cax1)
    fig.colorbar(axes[-1].images[0], cax=cax2)
    fig.subplots_adjust(left=0.165, right=0.895, top=0.925, bottom=0.075, hspace=0.10)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return metrics


def main() -> None:
    patch_cuda_for_cpu()
    torch.set_num_threads(4)
    np.random.seed(20260530)
    RECON_DIR.mkdir(parents=True, exist_ok=True)
    LATENT_DIR.mkdir(parents=True, exist_ok=True)

    selected = np.sort(np.random.default_rng(20260530).choice(np.arange(4900, 5000), size=5, replace=False))

    dpn_fno = load_dpn_reduction_model("FOAE_d50_kdv.pth")
    dpn_plain = load_dpn_reduction_model("OAE_d50_kdv.pth")
    ae = load_ae_reduction_model("AE_d50_kdv.pth")

    latent_fno = load_dm_latent_model("FOAE_d50_kdv_latent.pt")
    latent_plain = load_dm_latent_model("OAE_d50_kdv_latent.pt")
    latent_ae = load_ae_latent_model("AE_d50_kdv_latent.pt")

    with h5py.File(DM_DATA, "r") as f_dm, h5py.File(AE_DATA, "r") as f_ae:
        clean = f_dm["kdv_5000_501_512"]
        noisy = f_dm["kdv_5000_501_512"]
        latent_dpn_fno = f_dm["latent_DPN_d50_fno"]
        latent_dpn_plain = f_dm["latent_DPN_d50"]
        trunk_dpn_fno = f_dm["trunk_DPN_d50_fno"][:]
        trunk_dpn_plain = f_dm["trunk_DPN_d50"][:]
        latent_ae_data = f_ae["latent_fno_d50_m20_w1"]

        manifest = {
            "selected_sample_indices_zero_based": selected.tolist(),
            "selected_sample_numbers_one_based": (selected + 1).tolist(),
            "row_order": [
                "benchmark",
                "F-OAE",
                "OAE",
                "AE",
                "|F-OAE - benchmark|",
                "|OAE - benchmark|",
                "|AE - benchmark|",
            ],
            "reconstruction": [],
            "latent_prediction": [],
        }

        for idx in selected:
            y = clean[idx].astype(np.float32)
            x = noisy[idx].astype(np.float32)

            rec_f = reconstruct_dpn(dpn_fno, x)
            rec_o = reconstruct_dpn(dpn_plain, x)
            rec_a = reconstruct_ae(ae, x)
            rec_metrics = plot_panel(
                RECON_DIR / f"sample_{idx:04d}_reconstruction_rows7.png",
                y,
                rec_f,
                rec_o,
                rec_a,
                f"KdV reconstruction sample {idx}",
            )
            rec_metrics["sample_index_zero_based"] = int(idx)
            manifest["reconstruction"].append(rec_metrics)

            pred_f = predict_dpn_latent(latent_fno, latent_dpn_fno[idx, 0], trunk_dpn_fno, y.shape[0])
            pred_o = predict_dpn_latent(latent_plain, latent_dpn_plain[idx, 0], trunk_dpn_plain, y.shape[0])
            pred_a = predict_ae_latent(latent_ae, ae, latent_ae_data[idx, 0], y.shape[0])
            pred_metrics = plot_panel(
                LATENT_DIR / f"sample_{idx:04d}_latent_prediction_rows7.png",
                y,
                pred_f,
                pred_o,
                pred_a,
                f"KdV latent prediction sample {idx}",
            )
            pred_metrics["sample_index_zero_based"] = int(idx)
            manifest["latent_prediction"].append(pred_metrics)

            print(
                json.dumps(
                    {
                        "sample": int(idx),
                        "reconstruction_mse": rec_metrics,
                        "latent_prediction_mse": pred_metrics,
                    }
                ),
                flush=True,
            )

    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({"output": str(OUT_DIR), "samples": selected.tolist()}))


if __name__ == "__main__":
    main()
