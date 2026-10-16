import torch
from torch import nn
import scipy.io as io
import numpy as np
import sys
from pathlib import Path
REDUCTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REDUCTION_DIR))
from checkpoint_utils import load_reduction_checkpoint
import h5py
import torch.nn.functional as F
import matplotlib.pyplot as plt

model = load_reduction_checkpoint(REDUCTION_DIR / 'models' / 'AE_d150_ns.pth', 'ns', 'AE')
model.eval()

b = 5000
T = 161
S = 64
file_path = REDUCTION_DIR / 'data' / 'data_ns.h5df'
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
