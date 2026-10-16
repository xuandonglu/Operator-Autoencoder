from pathlib import Path
CASE_DIR = Path(__file__).resolve().parent
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import h5py
from timeit import default_timer
from utilities3 import UnitGaussianNormalizer
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA

def derivative_x(u):
    dx = 1 / 512
    ux = torch.zeros_like(u)
    ux[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / 2 / dx
    uxxx = torch.zeros_like(u)
    uxxx[..., 2:-2] = (-u[..., :-4] + 2 * u[..., 1:-3] - 2 * u[..., 3:-1] + u[..., 4:]) / (2 * dx ** 3)
    return ux, uxxx

def derivative_t(u):
    dt = 5 / 501
    ut = torch.zeros_like(u)
    ut[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / 2 / dt
    return ut

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.fc0 = nn.Linear(3, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape).cuda()
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.squeeze(-1)

    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

################################################################
# configs
################################################################
if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    # time.sleep(3300)

    ntrain = 4900
    ntest = 100
    modes = 10
    width = 24
    batch_size = 4
    epochs = 100
    learning_rate = 0.001
    model = FNO(6, modes, width).cuda()
    model.register_buffer('train_grid', torch.arange(0, 501, step=50))
    reduction_model = 'DPN_d10'
    pi_weight = '1e-2'
    path_model = 'model/kdv1d_' + f'N{ntrain}_' + \
                 f'p{len(model.train_grid)}' + f'_m{model.modes1}' + f'_{pi_weight}pi' + f'_{reduction_model}'
    path_model = CASE_DIR / path_model
    path_model.parent.mkdir(parents=True, exist_ok=True)

    ################################################################
    # load data
    ################################################################
    with h5py.File(CASE_DIR.parents[1] / 'reduction' / 'data' / 'data_kdv.h5df', 'r') as f:
        u = f['kdv_5000_501_512'][:]
        u_latent = f[f'latent_{reduction_model}'][:]
        tr = f[f'trunk_{reduction_model}'][:]
    u = torch.from_numpy(u).float()
    u_latent = torch.from_numpy(u_latent).float()
    x_normalizer = UnitGaussianNormalizer(u_latent[:1000, 0, :].unsqueeze(-1))
    y_normalizer = UnitGaussianNormalizer(u_latent[:1000])
    T = u.shape[1]
    S = u.shape[2]
    Z = u_latent.shape[-1]

    train_x = u_latent[:ntrain, 0]
    train_y = u_latent[:ntrain]
    test_x = u_latent[-ntest:, 0]
    test_y = u[-ntest:]
    train_x = train_x.reshape(ntrain, 1, train_x.shape[1], 1).repeat([1, T, 1, 1])
    test_x = test_x.reshape(ntest, 1, test_x.shape[1], 1).repeat([1, T, 1, 1])

    # train_x = x_normalizer.encode(train_x)
    # test_x = x_normalizer.encode(test_x)
    # train_y = y_normalizer.encode(train_y)
    # y_normalizer.cuda()

    train_y = train_y[:, model.train_grid]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=batch_size)
    device = torch.device('cuda')
    tr = torch.from_numpy(tr).float().to(device)

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=epochs, div_factor=2,
                                                    final_div_factor=1e-3 / 1e-6, pct_start=0.2)
    MSELoss = nn.MSELoss(reduction='mean').cuda()
    for ep in range(epochs):
        t1 = default_timer()
        model.train()
        train_mse = 0
        train_lp = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x.permute(0, 2, 1, 3)).permute(0, 2, 1)
            mse = MSELoss(out[:, model.train_grid], y)
            # out = y_normalizer.decode(out)
            full_out = torch.einsum("bti,io->bto", out, tr)
            ut = derivative_t(full_out)[:, 1:-1, 2:-2]
            ux, uxxx = derivative_x(full_out)
            orig = full_out[:, 1:-1, 2:-2]
            ux = ux[:, 1:-1, 2:-2]
            uxxx = uxxx[:, 1:-1, 2:-2]
            lp = MSELoss(ut + 6 * orig * ux, -0.001 * uxxx)
            loss = mse + float(pi_weight) * lp
            loss.backward()
            optimizer.step()
            train_mse += mse.item() / ntrain * batch_size
            train_lp += lp.item() / ntrain * batch_size
        scheduler.step()
        t2 = default_timer()
        t_train = t2 - t1
        print(f"{ep:3d}, Train_mse: {train_mse:.4e}, Train_lp: {train_lp:.4e}, time train: {t_train:2.3f}")

        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                t1 = default_timer()
                test_loss = 0
                for x, y in test_loader:
                    x, y = x.cuda(), y.cuda()
                    out = model(x.permute(0, 2, 1, 3)).permute(0, 2, 1)
                    # out = y_normalizer.decode(out)
                    out = torch.einsum("bti,io->bto", out, tr)
                    loss = MSELoss(out, y)
                    test_loss += loss.item() / ntest * batch_size
                t2 = default_timer()
                t_test = t2 - t1
            print(f"{ep:3d}, Test_loss: {test_loss:.4e}, time test: {t_test:2.3f}")
            torch.save(model, path_model)
            print('Model saving: ', path_model)
