import torch
from torch import nn
import sys
import numpy as np
from pathlib import Path
REDUCTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REDUCTION_DIR))
from checkpoint_utils import load_reduction_checkpoint
import h5py
import matplotlib.pyplot as plt
import scipy.io as io

MSELoss = nn.MSELoss(reduction='mean')
def loss_function(reconstruction, data):

    utt, uxx = derivative(data)
    utt1, uxx1 = derivative(reconstruction)
    mse = MSELoss(reconstruction, data)
    msexx = MSELoss(uxx, uxx1)
    msett = MSELoss(utt, utt1)
    for time_idx in range(0, min(utt.shape[1], uxx.shape[1]), 25):
        plt.plot(utt[0, time_idx, 1:-1].detach())
        plt.plot(utt1[0, time_idx, 1:-1].detach())
        plt.plot(uxx[0, time_idx, 1:-1].detach())
        plt.plot(uxx1[0, time_idx, 1:-1].detach())
        plt.title(time_idx)
        plt.show()
    return mse, msett, msexx

def derivative(u):
    dt = 10 / 501
    dx = 10 / 256
    uxx = (u[:, :, 2:] - 2 * u[:, :, 1:-1] + u[:, :, :-2]) / dx ** 2
    utt = (u[:, 2:, :] - 2 * u[:, 1:-1, :] + u[:, :-2, :]) / dt ** 2
    return utt, uxx

def pde_residual(u):
    utt, uxx = derivative(u)
    utt = utt[..., 1:-1]
    uxx = uxx[:, 1:-1]
    e = utt - uxx
    return e

def rrmse(prediction, target):
    numerator = torch.sum((prediction - target) ** 2)
    denominator = torch.sum(target ** 2)
    return torch.sqrt(numerator / denominator)

def rrmse_utt(u, u1):
    utt, _ = derivative(u)
    utt1, _ = derivative(u1)
    e = rrmse(utt, utt1)
    return e

model = load_reduction_checkpoint(REDUCTION_DIR / 'models' / 'AE_d30_wave.pth', 'wave', 'AE')
model.eval()

b = 1
T = 501
S = 256
with h5py.File(REDUCTION_DIR / 'data' / 'data_wave.h5df', 'r') as f:
    u = f['wave_5000_501_256'][:]
u = u[-b:, ...]
u = np.reshape(u, [u.shape[0] * u.shape[1], 1, u.shape[-1]])
u_test = torch.from_numpy(u.astype(np.float32))

u_rec = torch.zeros_like(u_test)
for i in range(u_test.shape[0]):
    u_rec[i] = model(u_test[i:i+1, ...].cuda()).cpu()
u_rec = u_rec.reshape([b, T, S])
u_test = u_test.reshape([b, T, S])
loss, losstt, lossx = loss_function(u_rec, u_test)
loss = rrmse(u_rec, u_test)
loss = loss.detach().cpu().numpy()
losstt = rrmse_utt(u_rec, u_test)
losstt = losstt.detach().cpu().numpy()

e_rec = pde_residual(u_rec)
e0 = torch.zeros_like(e_rec)
error = np.sqrt((torch.sum(e_rec**2) / torch.sum(u_rec[:, 1:-1, 1:-1]**2)).detach().cpu().numpy())
u_test_np = u_test.cpu().detach().numpy()
u_rec_np = u_rec.cpu().detach().numpy()
e_rec_np = e_rec.cpu().detach().numpy()
io.savemat('fno_rec.mat', {'u_rec': u_rec_np[0], 'e_rec': e_rec_np[0], 'u_test': u_test_np[0]})

# for i in range(0, 20):
#     test_sample = u_test_np[0, 10*i, :]
#     rec_sample = u_rec_np[0, 10*i, :]
#     plt.title(i*10)
#     plt.plot(test_sample, label='True')
#     plt.plot(rec_sample, label='Reconstructed')
#     plt.legend()
#     plt.show()

