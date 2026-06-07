import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

THIS_DIR = Path(__file__).resolve().parent
REDUCTION_DIR = THIS_DIR.parent
sys.path.append(str(REDUCTION_DIR))
import nn_ae_cylinder

from load_test_utils import load_last_flat_samples


MSELoss = nn.MSELoss(reduction='mean')


def loss_function(reconstruction, data):
    return MSELoss(reconstruction, data)


model_name = 'AE_d140_cylinder.pth'
model_path = THIS_DIR / model_name
data_path = REDUCTION_DIR / 'data' / 'data_cylinder.h5df'
dataset_name = 'uxuy_50_300_1_768_512'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(model_path, weights_only=False, map_location=device).to(device)
model.eval()

b = 300
u = load_last_flat_samples(data_path, dataset_name, b)
u = torch.from_numpy(u.astype(np.float32))

u_rec = torch.empty_like(u)
with torch.no_grad():
    bs = 1
    for i in range(0, u.shape[0], bs):
        pred = model(u[i:i + bs].to(device))
        u_rec[i:i + bs] = pred.cpu()

loss = loss_function(u_rec, u)
loss = loss.cpu().detach().numpy()
print(f"Test_loss: {loss:.2e}")

u_np = u.cpu().detach().numpy()
u_rec_np = u_rec.cpu().detach().numpy()
for i in range(b):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmin = min(u_np[i].min(), u_rec_np[i].min())
    vmax = max(u_np[i].max(), u_rec_np[i].max())
    im1 = axes[0].imshow(u_np[i, 0], origin='lower', cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
    im2 = axes[1].imshow(u_rec_np[i, 0], origin='lower', cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(im1, ax=axes, shrink=0.8)
    plt.show()
