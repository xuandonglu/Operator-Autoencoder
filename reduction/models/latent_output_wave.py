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
# time.sleep(5500)

model_name = "DPN_d30.pth"
name_without_ext = model_name.rsplit(".pth", 1)[0]
model = load_reduction_checkpoint(REDUCTION_DIR / 'models' / 'OAE_d30_wave.pth', 'wave', 'OAE')
model.eval()

file_path = REDUCTION_DIR / 'data' / 'data_wave.h5df'
with h5py.File(file_path, 'r') as f:
    u = f['wave_5000_501_256'][:]
b = u.shape[0]
T = u.shape[-1]
S = u.shape[1]
u = np.reshape(u, [u.shape[0] * u.shape[1], u.shape[-1]])
u_test = torch.from_numpy(u.astype(np.float32))

u_latent = np.zeros((b, S, model.latent_dim))
u_rec = np.zeros((b, S, T))
batch_size = u_rec.shape[1]
for i in range(u_rec.shape[0]):
    single_sample = u_test[i*batch_size:(i+1)*batch_size, ...]
    z = model.encode(single_sample.cuda())
    u_latent[i, ...] = z.view(S, model.latent_dim).detach().cpu().numpy().astype(np.float32)

dataset = f'latent_{name_without_ext}'
with h5py.File(file_path, 'a') as hdf:
    if dataset in hdf:
        hdf[dataset][...] = u_latent.astype('float32')
    else:
        hdf.create_dataset(dataset, data=u_latent, dtype='float32')

trunk_out = model.trunk(next(model.parameters()).device).detach().cpu().permute(1, 0).numpy().astype(np.float32)
dataset_trunk = f'trunk_{name_without_ext}'
with h5py.File(file_path, 'a') as hdf:
    if dataset_trunk in hdf:
        hdf[dataset_trunk][...] = trunk_out.astype('float32')
    else:
        hdf.create_dataset(dataset_trunk, data=trunk_out, dtype='float32')
