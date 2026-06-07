import torch
from torch import nn
import sys
import numpy as np
sys.path.append('../')
from nn import DPN
import h5py
import matplotlib.pyplot as plt
import torch.fft as fft

MSELoss = nn.MSELoss(reduction='mean')

def loss_function(rec, data):
    mse = MSELoss(rec, data)
    wt1, du1 = derivative(data)
    wt2, du2 = derivative(rec)
    mse1 = MSELoss(wt1, wt2)
    mse2 = MSELoss(du1, du2)
    # for i in range(16):
    #     j = 10
    #     plt.plot(wt1[0, i * 10, j, :].detach())
    #     plt.plot(wt2[0, i * 10, j, :].detach())
    #     plt.title(i * 10)
    #     plt.show()
    #     plt.plot(du1[0, i*10, j, :].detach())
    #     plt.plot(du2[0, i*10, j, :].detach())
    #     plt.title(i*10)
    #     plt.show()
    #     plt.plot(rhs1[0, i*k, j, :].detach())
    #     plt.plot(rhs2[0, i*k, j, :].detach())
    #     plt.title(i*k)
    #     plt.show()
    return mse, mse1, mse2

def derivative(u):
    u = u.permute(0, 1, 3, 2)
    N = 64
    nu = 0.001
    dt = 0.25
    wt = torch.zeros_like(u)
    wt[:, 1:-1] = (u[:, 2:, :, :] - u[:, :-2, :, :]) / (2 * dt)
    k = torch.fft.fftfreq(N, d=1.0 / N).to(u.device)
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    lap = -(2 * torch.pi) ** 2 * (kx ** 2 + ky ** 2)
    lap[0, 0] = 1.0
    wh = fft.fft2(u)
    psi_h = wh / lap
    ux = torch.real(fft.ifft2(1j * 2 * torch.pi * ky * psi_h))
    uy = torch.real(fft.ifft2(-1j * 2 * torch.pi * kx * psi_h))
    wx = torch.real(fft.ifft2(1j * 2 * torch.pi * kx * wh))
    wy = torch.real(fft.ifft2(1j * 2 * torch.pi * ky * wh))
    d2w = nu * torch.real(fft.ifft2(lap * wh))
    lhs = wt + ux * wx + uy * wy
    rhs = d2w + f
    du = ux * wx + uy * wy - d2w
    return wt, du

def pde_residual(u):
    wt, du = derivative(u)
    e = du + wt - f
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

model = torch.load('DPN_d150_fno.pth', weights_only=False)
model.eval()

b = 1
with h5py.File('../data/data.hdf5', 'r') as f:
    u_clean = f['ns_5000_161_64_64'][:][-b:]
    f = f['f'][:]
f = torch.from_numpy(f.astype(np.float32)).unsqueeze(0)
T = u_clean.shape[1]
S = u_clean.shape[-1]
u_clean = np.reshape(u_clean, [b * T, S, S])
u_clean = torch.from_numpy(u_clean.astype(np.float32))
u_rec = torch.zeros_like(u_clean)
for i in range(b):
    u_rec[i*T:(i+1)*T] = model(u_clean[i*T:(i+1)*T, ...].cuda()).cpu()
u_rec = u_rec.reshape([b, T, S, S])
u_clean = np.reshape(u_clean, [b, T, S, S])

loss, loss1, loss2 = loss_function(u_rec, u_clean)
loss = loss.cpu().detach().numpy()
e_rec = pde_residual(u_rec)
e0 = torch.zeros_like(e_rec)
error = MSELoss(e_rec, e0).cpu().detach().numpy()
print(f"Test_loss: {loss:.2e}, pde_residual: {error:.2e}")

# u_clean_np = u_clean[:, 1:-1, 1:-1].cpu().detach().numpy()
# u_rec_np = u_rec[:, 1:-1, 1:-1].cpu().detach().numpy()
# e_rec_np = e_rec[:, 1:-1, 1:-1].cpu().detach().numpy()
# for i in range(b):
#     for j in range(16):
#         test_sample = u_clean_np[i, 10*j, 10, 1:-1]
#         rec_sample = u_rec_np[i, 10*j, 10, 1:-1]
#         e_sample = e_rec_np[i, 10*j, 10, 1:-1]
#         plt.plot(test_sample, label='True')
#         plt.plot(rec_sample, label='Reconstructed')
#         plt.plot(e_sample)
#         plt.legend()
#         plt.show()
