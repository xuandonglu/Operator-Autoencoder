import torch
from torch import nn
import scipy.io as io
import numpy as np
import sys
sys.path.append('../')
from nn import AE as ae
import h5py
import torch.nn.functional as F
import matplotlib.pyplot as plt

model = torch.load("fno_d150_m12_w1.pth", weights_only=False)
model.eval()

b = 5000
T = 161
S = 64
file_path = '../data/data.hdf5'
with h5py.File(file_path, 'r') as f:
    u = f['ns_noise_5000_161_64_64'][:]
train_grid = np.linspace(0, T-1, num=T).astype(int)
u_latent = np.zeros((b, T, model.latent_dim))
for i in range(b):
    single_sample = torch.from_numpy(u[i, train_grid, ...]).unsqueeze(1)
    z = model.encode(single_sample.cuda())
    u_latent[i, ...] = z.view(T, model.latent_dim).detach().cpu().numpy().astype(np.float32)

dataset = 'u_latent_fno_d150_m12_w1'
with h5py.File(file_path, 'a') as hdf:
    if dataset in hdf:
        hdf[dataset][...] = u_latent.astype('float32')
    else:
        hdf.create_dataset(dataset, data=u_latent, dtype='float32')