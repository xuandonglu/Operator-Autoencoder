import torch
import numpy as np
import sys
from pathlib import Path
REDUCTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REDUCTION_DIR))
from checkpoint_utils import load_reduction_checkpoint
import h5py
import matplotlib.pyplot as plt
import time
# time.sleep(1000)

model_name = "DPN_d150.pth"
name_without_ext = model_name.rsplit(".pth", 1)[0]
model = load_reduction_checkpoint(REDUCTION_DIR / 'models' / 'OAE_d150_ns.pth', 'ns', 'OAE')
model.eval()

file_path = REDUCTION_DIR / 'data' / 'data_ns.h5df'
with h5py.File(file_path, 'r') as f:
    u = f['ns_noise_5000_161_64_64'][:]
b = u.shape[0]
T = u.shape[-1]
S = u.shape[1]
u = np.reshape(u, [u.shape[0] * u.shape[1], u.shape[-2], u.shape[-1]])
u_latent = np.zeros((b, S, model.latent_dim))
for i in range(b):
    u_latent[i, ...] = model.encode(torch.from_numpy(u[i*S:(i+1)*S, ...]).cuda()).view(S, model.latent_dim).detach().cpu().numpy().astype(np.float32)
    print('Sample ', i)

dataset = f'latent_{name_without_ext}'
with h5py.File(file_path, 'a') as hdf:
    if dataset in hdf:
        hdf[dataset][...] = u_latent.astype('float32')
    else:
        hdf.create_dataset(dataset, data=u_latent, dtype='float32')

trunk_out = model.trunk(next(model.parameters())).detach().cpu().permute(2, 0, 1).numpy().astype(np.float32)
dataset_trunk = f'trunk_{name_without_ext}'
with h5py.File(file_path, 'a') as hdf:
    if dataset_trunk in hdf:
        hdf[dataset_trunk][...] = trunk_out.astype('float32')
    else:
        hdf.create_dataset(dataset_trunk, data=trunk_out, dtype='float32')
