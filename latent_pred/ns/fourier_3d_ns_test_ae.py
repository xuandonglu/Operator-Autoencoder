from pathlib import Path
CASE_DIR = Path(__file__).resolve().parent
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import h5py
from timeit import default_timer
import matplotlib.pyplot as plt
import scipy.io as io
from fourier_3d_ns_ae import SpectralConv3d, FNO, derivative

ntest = 1
batch_size = 1
path_model = CASE_DIR / 'model' / 'ns2d_N4900_p9_1e0pi'

################################################################
# load data
################################################################
with h5py.File(CASE_DIR.parents[1] / 'reduction' / 'data' / 'data_ns.h5df', 'r') as f:
    u = f['ns_5000_161_64_64'][:]
    f = f['f'][:]
u = torch.from_numpy(u).float()[-5000:]
f = torch.from_numpy(f).float().unsqueeze(0).unsqueeze(0).cuda()
T = u.shape[1]
S = u.shape[2]

test_x = u[-ntest:, 0, :]
test_y = u[-ntest:, ...]
test_x = test_x.reshape(ntest, 1, S, S, 1).repeat([1, T, 1, 1, 1])

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=batch_size)
device = torch.device('cuda')
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
        out = model(x)
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
