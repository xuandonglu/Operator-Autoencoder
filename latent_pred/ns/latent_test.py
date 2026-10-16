import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import h5py
from timeit import default_timer
import matplotlib.pyplot as plt
import scipy.io as io
from latent import SpectralConv2d_fast, FNO, derivative_t, derivative
from pathlib import Path
CASE_DIR = Path(__file__).resolve().parent

ntest = 100
batch_size = 1
train_grid = torch.arange(0, 161, step=20)
path_model = CASE_DIR / 'models' / 'OAE_d150_ns_latent.pt'
reduction_model = 'DPN_d150'

################################################################
# load data
################################################################
with h5py.File(CASE_DIR.parents[1] / 'reduction' / 'data' / 'data_ns.h5df', 'r') as f:
    u = f['ns_5000_161_64_64'][:]
    u_latent = f[f'latent_{reduction_model}'][:]
    tr = f[f'trunk_{reduction_model}'][:]
u = torch.from_numpy(u).float()
u_latent = torch.from_numpy(u_latent).float()
T = u.shape[1]
S = u.shape[-1]
Z = u_latent.shape[-1]

test_x = u_latent[-ntest:, 0]
test_y = u[-ntest:]
test_x = test_x.reshape(ntest, 1, test_x.shape[1], 1).repeat([1, T, 1, 1])

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=batch_size)
device = torch.device('cuda')
tr = torch.from_numpy(tr).float().to(device).reshape([Z, S, S])
################################################################
# training and evaluation
################################################################
model = torch.load(path_model, weights_only=False)

def rrmse(prediction, target):
    numerator = torch.sum((prediction - target) ** 2)
    denominator = torch.sum(target ** 2)
    return torch.sqrt(numerator / denominator)

MSELoss = nn.MSELoss(reduction='mean').cuda()
with torch.no_grad():
    test_loss = 0
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        out = model(x.permute(0, 2, 1, 3)).permute(0, 2, 1)
        out = torch.einsum("bti,ixy->btxy", out, tr)
        loss = MSELoss(out, y)
        test_loss += loss.item() / ntest
        # io.savemat('output/palfno.mat', {'u': out[0].detach().cpu().numpy()})

        # for i in range(17):
        #     plt.plot(y[0, :, i*10, 10].detach().cpu())
        #     plt.plot(out[0, :, i*10, 10].detach().cpu())
        #     plt.scatter(train_grid, y[0, train_grid, i*10, 10].detach().cpu())
        #     plt.title(i*10)
        #     plt.show()

        # for i in range(20):
        #     plt.subplot(1, 2, 1)
        #     plt.contourf(y[0, i * 10, :, :].detach().cpu())
        #     plt.subplot(1, 2, 2)
        #     plt.contourf(out[0, i * 10, :, :].detach().cpu())
        #     plt.title(i * 10)
        #     plt.show()
print(f"Test_loss: {test_loss:.4e}")
