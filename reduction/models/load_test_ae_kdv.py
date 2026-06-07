import torch
from torch import nn
import sys
import numpy as np
sys.path.append('../../nd')
sys.path.append('../')
from nn import AE
import h5py
import matplotlib.pyplot as plt
import scipy.io as io
from dpn_derivative import DIFF, dudx, d3udx3

def rrmse(prediction, target):
    numerator = torch.sum((prediction - target) ** 2)
    denominator = torch.sum(target ** 2)
    return torch.sqrt(numerator / denominator)

MSELoss = nn.MSELoss(reduction='mean')
def loss_function(reconstruction, data):

    ut, ux, uxx, uxxx = derivative(data)
    ut1, ux1, uxx1, uxxx1 = derivative(reconstruction)
    mse = MSELoss(reconstruction, data)
    msexxx = MSELoss(uxxx, uxxx1)
    msexx = MSELoss(uxx, uxx1)
    msex = MSELoss(ux, ux1)
    # for i in range(12):
    #     plt.plot(uxx[0, i*12, :].detach())
    #     plt.plot(uxx1[0, i*12, :].detach())
    #     plt.title(i*12)
    #     plt.show()
    #     plt.plot(uxxx[0, i * 12, :].detach())
    #     plt.plot(uxxx1[0, i * 12, :].detach())
    #     plt.title(i * 12)
    #     plt.show()
    return mse, msex, msexx, msexxx

def derivative(u):
    ux = torch.zeros_like(u)
    uxx = torch.zeros_like(u)
    uxxx = torch.zeros_like(u)
    ut = torch.zeros_like(u)
    dx = 1 / u.shape[-1]
    ux[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / 2 / dx
    uxx[:, :, 1:-1] = (u[:, :, 2:] - 2 * u[:, :, 1:-1] + u[:, :, :-2]) / dx ** 2
    uxxx[:, :, 2:-2] = (-u[:, :, :-4] + 2 * u[:, :, 1:-3] - 2 * u[:, :, 3:-1] + u[:, :, 4:]) / (2 * dx**3)
    dt = 5 / u.shape[1]
    ut[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / 2 / dt
    return ut, ux, uxx, uxxx

def pde_residual_nd(u):
    ut, _, _, _ = derivative(u)
    ux = diff_x(u.reshape([b*T, S]).cuda()).reshape([b, T, S]).cpu()
    uxxx = diff_xxx(u.reshape([b*T, S]).cuda()).reshape([b, T, S]).cpu()
    e = ut + 6 * u * ux + uxxx
    return e

def pde_residual(u):
    ut, ux, uxx, uxxx = derivative(u)
    e = ut + 6 * u * ux + 0.001 * uxxx
    return e

def rrmse_uxxx(u, u1):
    _, _, _, uxxx = derivative(u)
    _, _, _, uxxx1 = derivative(u1)
    e = rrmse(uxxx, uxxx1)
    return e

model = torch.load("fno_d20_m20_w1.pth")
model.eval()
diff_xxx = torch.load("../../nd/dpn_uxxx_50_N900_p11")
diff_xxx.eval()
for param in diff_xxx.parameters():
    param.requires_grad = False

diff_x = torch.load("../../nd/dpn_ux_50_N900_p11")
diff_x.eval()
for param in diff_x.parameters():
    param.requires_grad = False

b = 100
T = 501
S = 512
with h5py.File('../data/data.hdf5', 'r') as f:
    u = f['kdv_5000_501_512'][:]
u = u[-b:, ...]
u = np.reshape(u, [u.shape[0] * u.shape[1], 1, u.shape[-1]])
u_test = torch.from_numpy(u.astype(np.float32))

u_rec = torch.zeros_like(u_test)
for i in range(b):
    u_rec[i*T:(i+1)*T] = model(u_test[i*T:(i+1)*T, ...].cuda()).cpu()
u_rec = u_rec.reshape([b, T, S])
u_test = u_test.reshape([b, T, S])
loss, lossx, lossxx, lossxxx = loss_function(u_rec, u_test)
# loss = rrmse(u_rec, u_test)
loss = loss.detach().cpu().numpy()
lossxxx = rrmse_uxxx(u_rec, u_test)
lossxxx = lossxxx.detach().cpu().numpy()

e_rec = pde_residual(u_rec)[:, 1:-1, 1:-1]
e0 = torch.zeros_like(e_rec)
error = MSELoss(e_rec, e0).detach().cpu().numpy()# / MSELoss(u_rec, e0)

u_test_np = u_test[:, 1:-1, 1:-1].cpu().detach().numpy()
u_rec_np = u_rec[:, 1:-1, 1:-1].cpu().detach().numpy()
e_rec_np = e_rec.cpu().detach().numpy()
io.savemat('fno_rec.mat', {'u_rec': u_rec_np[0], 'u_test': u_test_np[0], 'e_rec': e_rec_np[0]})

for i in range(0, 40):
    test_sample = u_test_np[0, 12*i, :]
    rec_sample = u_rec_np[0, 12*i, :]
    e_sample = e_rec_np[0, 12*i, :]
    plt.title(i*12)
    plt.plot(test_sample, label='True')
    plt.plot(rec_sample, label='Reconstructed')
    plt.plot(e_sample)
    plt.legend()
    plt.show()

