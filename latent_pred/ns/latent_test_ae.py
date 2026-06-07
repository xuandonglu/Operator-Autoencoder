import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utilities3 import *
import matplotlib.pyplot as plt
import scipy.io as io
from latent import SpectralConv2d_fast, FNO, derivative

ntest = 1
batch_size = 1
reduction_model = 'fno_d10_m12_w1'
path_model = f'model/ns2d_N4900_p17_1e0pi_{reduction_model}'
train_grid = torch.arange(0, 161, step=10)

################################################################
# load derivative models
################################################################
import sys
sys.path.append('../reduction')
from nn import AE
AE = torch.load(f"../reduction/models/{reduction_model}.pth")
AE.eval()
for param in AE.parameters():
    param.requires_grad = False

################################################################
# load data
################################################################
with h5py.File('../reduction/data/data.hdf5', 'r') as f:
    u = f['ns_5000_161_64_64'][:]
    u_latent = f[f'u_latent_{reduction_model}'][:]
    f = f['f'][:]
u = torch.from_numpy(u).float()[-5000:]
f = torch.from_numpy(f).float().unsqueeze(0).unsqueeze(0).cuda()
u_latent = torch.from_numpy(u_latent).float()
x_normalizer = UnitGaussianNormalizer(u_latent[:1000, 0, :].unsqueeze(-1))
y_normalizer = UnitGaussianNormalizer(u_latent[:1000])
T = u.shape[1]
S = u.shape[-1]
Z = u_latent.shape[-1]

test_x = u_latent[-ntest:, 0]
test_y = u[-ntest:]
test_x = test_x.reshape(ntest, 1, test_x.shape[1], 1).repeat([1, T, 1, 1])

test_x = x_normalizer.encode(test_x)
y_normalizer.cuda()

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=batch_size)
device = torch.device('cuda')
################################################################
# training and evaluation
################################################################
model = torch.load(path_model)

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
        out = AE.decode(y_normalizer.decode(out).reshape([batch_size * T, Z])).reshape([batch_size, T, S, S])
        loss = MSELoss(out, y)
        test_loss += loss.item() / ntest
        io.savemat(f'{path_model}.mat', {'out': out[-1].cpu().detach().numpy(), 'y': y[-1].cpu().detach().numpy()})

        # for i in range(20):
        #     plt.subplot(1, 2, 1)
        #     plt.contourf(y[0, i * 10, :, :].detach().cpu())
        #     plt.subplot(1, 2, 2)
        #     plt.contourf(out[0, i * 10, :, :].detach().cpu())
        #     plt.title(i * 10)
        #     plt.show()
print(f"Test_loss: {test_loss:.4e}")
