import torch
from torch import nn
import sys
import numpy as np
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
MODELS_DIR = THIS_FILE.parent
REDUCTION_DIR = MODELS_DIR.parent
sys.path.insert(0, str(REDUCTION_DIR))

import nn_wave  # noqa: E402
from nn_wave import DPN  # noqa: E402

# Older checkpoints were pickled when the model module was named "nn".
sys.modules.setdefault('nn', nn_wave)
globals()['DPN'] = nn_wave.DPN
globals()['FNO1'] = nn_wave.FNO1
globals()['FNO2'] = nn_wave.FNO2
globals()['ResidualBlock'] = nn_wave.ResidualBlock
globals()['SpectralConv1d'] = nn_wave.SpectralConv1d
import h5py
import matplotlib.pyplot as plt

MSELoss = nn.MSELoss(reduction='mean')

def loss_function(reconstruction, data):
    utt, uxx = derivative(data)
    utt1, uxx1 = derivative(reconstruction)
    mse = MSELoss(reconstruction, data)
    msexx = MSELoss(uxx, uxx1)
    msett = MSELoss(utt, utt1)
    # for i in range(51):
    #     plt.plot(uxx[0, i * 10, :].detach())
    #     plt.plot(uxx1[0, i * 10, :].detach())
    #     plt.title(f'uxx_{i}')
    #     plt.show()
    #     plt.plot(ut[0, :, i * 10].detach())
    #     plt.plot(ut1[0, :, i * 10].detach())
    #     plt.title(f'utt_{i}')
    #     plt.show()
    return mse, msexx, msett

def rrmse(prediction, target):
    numerator = torch.sum((prediction - target) ** 2)
    denominator = torch.sum(target ** 2)
    return torch.sqrt(numerator / denominator)

def derivative(u):
    dx = 10 / u.shape[-1]
    uxx = torch.zeros_like(u)
    uxx[:, :, 1:-1] = (u[:, :, 2:] - 2 * u[:, :, 1:-1] + u[:, :, :-2]) / dx ** 2
    uxx[:, :, 0] = (2 * u[:, :, 0] - 5 * u[:, :, 1] + 4 * u[:, :, 2] - u[:, :, 3]) / dx ** 2
    uxx[:, :, -1] = (2 * u[:, :, -1] - 5 * u[:, :, -2] + 4 * u[:, :, -3] - u[:, :, -4]) / dx ** 2
    dt = 10 / u.shape[-2]
    utt = torch.zeros_like(u)
    utt[:, 1:-1, :] = (u[:, 2:, :] - 2 * u[:, 1:-1, :] + u[:, :-2, :]) / dt ** 2
    utt[:, 0, :] = (2 * u[:, 0, :] - 5 * u[:, 1, :] + 4 * u[:, 2, :] - u[:, 3, :]) / dt ** 2
    utt[:, -1, :] = (2 * u[:, -1, :] - 5 * u[:, -2, :] + 4 * u[:, -3, :] - u[:, -4, :]) / dt ** 2
    ut = torch.zeros_like(u)
    ut[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / 2 / dt
    return utt, uxx

def pde_residual(u):
    utt, uxx = derivative(u)
    utt = utt[:, 1:-1, 1:-1]
    uxx = uxx[:, 1:-1, 1:-1]
    e = utt - uxx
    return e

def derivative_x(u):
    dx = 10 / u.shape[-1]
    uxx = torch.zeros_like(u)
    uxx[..., 1:-1] = (u[..., 2:] - 2 * u[..., 1:-1] + u[..., :-2]) / dx ** 2
    uxx[..., 0] = (2 * u[..., 0] - 5 * u[..., 1] + 4 * u[..., 2] - u[..., 3]) / dx ** 2
    uxx[..., -1] = (2 * u[..., -1] - 5 * u[..., -2] + 4 * u[..., -3] - u[..., -4]) / dx ** 2
    return uxx

def derivative_t(u):
    dt = 10 / 501
    utt = torch.zeros_like(u)
    utt[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dt ** 2
    utt[:, 0] = (2 * u[:, 0] - 5 * u[:, 1] + 4 * u[:, 2] - u[:, 3]) / dt ** 2
    utt[:, -1] = (2 * u[:, -1] - 5 * u[:, -2] + 4 * u[:, -3] - u[:, -4]) / dt ** 2
    return utt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = MODELS_DIR / 'FOAE_d30_wave.pth'
model = torch.load(model_path, map_location=device, weights_only=False).to(device)
model.eval()

b = 100
with h5py.File(REDUCTION_DIR / 'data' / 'data_wave.h5df', 'r') as f:
    u_clean = f['wave_5000_501_256'][:][-b:]
    u = f['wave_5000_501_256'][:][-b:]
T = u_clean.shape[1]
S = u_clean.shape[2]
u_clean = np.reshape(u_clean, [b * T, S])
u = np.reshape(u, [b * T, S])
u = torch.from_numpy(u.astype(np.float32))
u_clean = torch.from_numpy(u_clean.astype(np.float32))
u_rec = torch.zeros_like(u)
with torch.no_grad():
    for i in range(b):
        single_sample = u[i*T:(i+1)*T, ...].to(device)
        u_rec[i*T:(i+1)*T] = model(single_sample).detach().cpu()
u_rec = u_rec.reshape([b, T, S])
u_clean = np.reshape(u_clean, [b, T, S])
u = np.reshape(u, [b, T, S])

loss, lossxx, losstt = loss_function(u_rec, u_clean)
# loss = rrmse(u_rec, u_clean)
loss = loss.cpu().detach().numpy()
e_rec = pde_residual(u_rec)
e0 = torch.zeros_like(e_rec)
e = MSELoss(e_rec, e0)

u_clean_np = u_clean[:, 1:-1, 2:-2].cpu().detach().numpy()
u_rec_np = u_rec[:, 1:-1, 2:-2].cpu().detach().numpy()
e_rec_np = e_rec[:, 1:-1, 2:-2].cpu().detach().numpy()
for i in range(b):
    for j in range(40):
        clean_sample = u_clean_np[i, j*5, :]
        rec_sample = u_rec_np[i, j*5, :]
        e_sample = e_rec_np[i, j*5, :]
        plt.plot(clean_sample, label='True')
        plt.plot(rec_sample, label='Reconstructed')
        plt.plot(e_sample, label='PDE residual')
        plt.title(j*5)
        plt.legend()
        plt.show()
    for j in range(50):
        clean_sample = u_clean_np[i, :, j*10]
        rec_sample = u_rec_np[i, :, j*10]
        e_sample = e_rec_np[i, :, j*10]
        plt.plot(clean_sample, label='True')
        plt.plot(rec_sample, label='Reconstructed')
        # plt.plot(e_sample, label='PDE residual')
        plt.legend()
        plt.show()
