import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import h5py
from timeit import default_timer
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from latent import SpectralConv2d_fast, FNO

ntest = 1
batch_size = 1
path_model = 'model/kdv1d_N4900_p11_m26_1e-3pi_DPN_d50'
reduction_model = path_model.rsplit("DPN", 1)[1]

################################################################
# load data
################################################################
with h5py.File('../reduction/data/data.hdf5', 'r') as f:
    u = f['kdv_5000_501_512'][:]
    u_latent = f[f'latent_DPN{reduction_model}'][:]
    tr = f[f'trunk_DPN{reduction_model}'][:]
u = torch.from_numpy(u).float()
u_latent = torch.from_numpy(u_latent).float()
T = u.shape[1]
S = u.shape[2]
Z = u_latent.shape[-1]

test_x = u_latent[-ntest:, 0]
test_y = u[-ntest:]
test_x = test_x.reshape(ntest, 1, test_x.shape[1], 1).repeat([1, T, 1, 1])

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=batch_size)
device = torch.device('cuda')
tr = torch.from_numpy(tr).float().to(device)

model = torch.load(path_model, weights_only=False)
MSELoss = nn.MSELoss(reduction='mean').cuda()
def rrmse(prediction, target):
    numerator = torch.sum((prediction - target) ** 2)
    denominator = torch.sum(target ** 2)
    return torch.sqrt(numerator / denominator)

with torch.no_grad():
    test_loss = 0
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        out = model(x.permute(0, 2, 1, 3)).permute(0, 2, 1)
        out = torch.einsum("bti,io->bto", out, tr)
        loss = MSELoss(out, y)
        test_loss += loss.item() / ntest

        # for i in range(20):
        #     plt.plot(y[0, i * 25, :].detach().cpu())
        #     plt.plot(out[0, i * 25, :].detach().cpu())
        #     plt.title(i * 25)
        #     plt.show()
print(f"Test_loss: {test_loss:.4e}")
