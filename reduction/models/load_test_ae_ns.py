import torch
from torch import nn
import scipy.io as io
import sys
import numpy as np
from pathlib import Path
REDUCTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REDUCTION_DIR))
from checkpoint_utils import load_reduction_checkpoint
import h5py
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import torch.fft as fft

def rrmse(prediction, target):
    numerator = torch.sum((prediction - target) ** 2)
    denominator = torch.sum(target ** 2)
    return torch.sqrt(numerator / denominator)

MSELoss = nn.MSELoss(reduction='mean')
def loss_function(rec, data):
    mse = MSELoss(rec, data)
    lhs1, rhs1 = derivative(data)
    lhs2, rhs2 = derivative(rec)
    mse1 = MSELoss(lhs1, lhs2)
    mse2 = MSELoss(rhs1, rhs2)
    k = 1
    # for i in range(20):
    #     j = 20
    #     plt.plot(data[0, i * k, j, :].detach())
    #     plt.plot(rec[0, i * k, j, :].detach())
    #     plt.title(i * k)
    #     plt.show()
    #     plt.plot(lhs1[0, i*k, j, :].detach())
    #     plt.plot(lhs2[0, i*k, j, :].detach())
    #     plt.title(i*k)
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
    w_t = torch.zeros_like(u)
    w_t[:, 1:-1] = (u[:, 2:, :, :] - u[:, :-2, :, :]) / (2 * dt)
    k = torch.fft.fftfreq(N, d=1.0 / N).to(u.device)
    k_x, k_y = torch.meshgrid(k, k, indexing="ij")
    lap = -(2 * torch.pi) ** 2 * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0
    w_h = fft.fft2(u)
    psi_h = w_h / lap
    u_x = torch.real(fft.ifft2(1j * 2 * torch.pi * k_y * psi_h))
    u_y = torch.real(fft.ifft2(-1j * 2 * torch.pi * k_x * psi_h))
    w_x = torch.real(fft.ifft2(1j * 2 * torch.pi * k_x * w_h))
    w_y = torch.real(fft.ifft2(1j * 2 * torch.pi * k_y * w_h))
    d2w = nu * torch.real(fft.ifft2(lap * w_h))
    lhs = w_t + u_x * w_x + u_y * w_y
    rhs = d2w + f
    return lhs, rhs

def pde_residual(u):
    lhs, rhs = derivative(u)
    e = lhs - rhs
    return e

model = load_reduction_checkpoint(REDUCTION_DIR / 'models' / 'AE_d150_ns.pth', 'ns', 'AE')
model.eval()

b = 1
with h5py.File(REDUCTION_DIR / 'data' / 'data_ns.h5df', 'r') as f:
    u_clean = f['ns_5000_161_64_64'][:][-b:]
    f = f['f'][:]
f = torch.from_numpy(f.astype(np.float32)).unsqueeze(0)
T = u_clean.shape[1]
S = u_clean.shape[-1]
u_clean = np.reshape(u_clean, [b * T, 1, S, S])
u_clean = torch.from_numpy(u_clean.astype(np.float32))
u_rec = torch.zeros_like(u_clean)
batch = T
for i in range(int(u_clean.shape[0]/batch)):
    u_rec[i*batch:(i+1)*batch] = model(u_clean[i*batch:(i+1)*batch, ...].cuda()).detach().cpu()
u_rec = u_rec.reshape([b, T, S, S])
u_clean = np.reshape(u_clean, [b, T, S, S])
loss, lossx, lossxx = loss_function(u_rec, u_clean)
loss = loss.detach().cpu().numpy()

e_rec = pde_residual(u_rec)
e0 = torch.zeros_like(e_rec)
error = MSELoss(e_rec, e0).detach().cpu().numpy()
print(f"Test_loss: {loss:.2e}, pde_residual: {error:.2e}")

# u_test_np = u_clean[:, 1:-1, 1:-1].cpu().detach().numpy()
# u_rec_np = u_rec[:, 1:-1, 1:-1].cpu().detach().numpy()
# e_rec_np = e_rec.cpu().detach().numpy()
# # io.savemat('fno_rec.mat', {'u_rec': u_rec_np[0], 'u_test': u_test_np[0], 'e_rec': e_rec_np[0]})
#
# for i in range(b):
#     for j in range(16):
#         test_sample = u_test_np[0, 10, j, :]
#         rec_sample = u_rec_np[0, 10, j, :]
#         e_sample = e_rec[0, 10, j, :]
#         plt.plot(test_sample, label='True')
#         plt.plot(rec_sample, label='Reconstructed')
#         plt.plot(e_sample)
#         plt.legend()
#         plt.show()
