from pathlib import Path
CASE_DIR = Path(__file__).resolve().parent
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
import scipy.io as io
from latent_ae import SpectralConv2d_fast, FNO, derivative

ntest = 1
batch_size = 1
reduction_model = 'fno_d150_m12_w1'
path_model = CASE_DIR / 'models' / 'AE_d150_ns_latent.pt'
train_grid = torch.arange(0, 161, step=10)

################################################################
# load derivative models
################################################################
import sys
sys.path.insert(0, str(CASE_DIR.parents[1] / 'reduction'))
from checkpoint_utils import load_reduction_checkpoint
AE = load_reduction_checkpoint(CASE_DIR.parents[1] / 'reduction' / 'models' / 'AE_d150_ns.pth', 'ns', 'AE')
AE.eval()
for param in AE.parameters():
    param.requires_grad = False

################################################################
# load data
################################################################
with h5py.File(CASE_DIR.parents[1] / 'reduction' / 'data' / 'data_ns.h5df', 'r') as f:
    u = f['ns_5000_161_64_64'][:]
    u_latent = f[f'u_latent_{reduction_model}'][:]
    f = f['f'][:]
u = torch.from_numpy(u).float()[-5000:]
f = torch.from_numpy(f).float().unsqueeze(0).unsqueeze(0).cuda()
u_latent = torch.from_numpy(u_latent).float()
T = u.shape[1]
S = u.shape[-1]
Z = u_latent.shape[-1]

test_x = u_latent[-ntest:, 0]
test_y = u[-ntest:]
test_x = test_x.reshape(ntest, 1, test_x.shape[1], 1).repeat([1, T, 1, 1])


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
        out = model(x.permute(0, 2, 1, 3)).permute(0, 2, 1)
        out = AE.decode(out.reshape([batch_size * T, Z])).reshape([batch_size, T, S, S])
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
