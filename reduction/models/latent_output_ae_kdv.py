import torch
import numpy as np
import sys
from pathlib import Path
REDUCTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REDUCTION_DIR))
from checkpoint_utils import load_reduction_checkpoint
import h5py
import matplotlib.pyplot as plt

model = load_reduction_checkpoint(REDUCTION_DIR / 'models' / 'AE_d50_kdv.pth', 'kdv', 'AE')
model.eval()

b = 5000
T = 501
S = 512
file_path = REDUCTION_DIR / 'data' / 'data_kdv.h5df'
with h5py.File(file_path, 'r') as f:
    u = f['kdv_5000_501_512'][:]
u_test = torch.from_numpy(u.astype(np.float32))
train_grid = np.linspace(0, T-1, num=T).astype(int)
u_latent = np.zeros((b, T, model.latent_dim))
for i in range(b):
    single_sample = u_test[i, train_grid, ...].unsqueeze(1)
    z = model.encode(single_sample.cuda()).cpu()
    u_latent[i, ...] = z.view(T,  model.latent_dim).detach().cpu().numpy().astype(np.float32)

dataset = 'latent_fno_d50_m20_w1'
with h5py.File(file_path, 'a') as hdf:
    if dataset in hdf:
        hdf[dataset][...] = u_latent.astype('float32')
    else:
        hdf.create_dataset(dataset, data=u_latent, dtype='float32')
