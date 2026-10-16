from pathlib import Path
CASE_DIR = Path(__file__).resolve().parent
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utilities3 import count_params
import h5py
from timeit import default_timer
import torch.fft as fft
import matplotlib.pyplot as plt
import time

def derivative(u):
    u = u.permute(0, 1, 3, 2)
    N = 64
    dt = 0.25
    nu = 0.001
    w_t = (u[:, 2:, :, :] - u[:, :-2, :, :]) / (2 * dt)
    w_curr = u[:, 1:-1, :, :]
    k = torch.fft.fftfreq(N, d=1.0 / N).to(u.device)
    k_x, k_y = torch.meshgrid(k, k, indexing="ij")
    lap = -(2 * torch.pi) ** 2 * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0
    w_h = fft.fft2(w_curr)
    psi_h = w_h / lap
    nl_term = torch.real(fft.ifft2(1j * 2 * torch.pi * k_y * psi_h)) * torch.real(fft.ifft2(1j * 2 * torch.pi * k_x * w_h))\
              + torch.real(fft.ifft2(-1j * 2 * torch.pi * k_x * psi_h)) * torch.real(fft.ifft2(1j * 2 * torch.pi * k_y * w_h))
    d2w = nu * torch.real(fft.ifft2(lap * w_h))
    return w_t + nl_term - d2w - f

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(4, self.width)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.squeeze(-1)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

################################################################
# configs
################################################################
if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    # time.sleep(2600)

    ntrain = 4900
    ntest = 100
    modes = 10
    width = 24
    batch_size = 4
    epochs = 100
    learning_rate = 0.001
    train_grid = torch.arange(0, 161, step=10)
    path_model = 'model/ns2d_' + f'N{ntrain}_' + f'p{len(train_grid)}' + '_1e0pi'
    path_model = CASE_DIR / path_model
    path_model.parent.mkdir(parents=True, exist_ok=True)

    ################################################################
    # load data
    ################################################################
    with h5py.File(CASE_DIR.parents[1] / 'reduction' / 'data' / 'data_ns.h5df', 'r') as f:
        u = f['ns_noise_5000_161_64_64'][:]
        f = f['f'][:]
    u = torch.from_numpy(u).float()[-5000:]
    f = torch.from_numpy(f).float().unsqueeze(0).unsqueeze(0).cuda()
    T = u.shape[1]
    S = u.shape[2]

    train_x = u[:ntrain, 0]
    train_y = u[:ntrain]
    test_x = u[:ntest, 0]
    test_y = u[:ntest]
    train_x = train_x.reshape(ntrain, 1, S, S, 1).repeat([1, T, 1, 1, 1])
    test_x = test_x.reshape(ntest, 1, S, S, 1).repeat([1, T, 1, 1, 1])
    train_y = train_y[:, train_grid]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=batch_size)
    device = torch.device('cuda')
    ################################################################
    # training and evaluation
    ################################################################
    model = FNO(modes, modes, modes, width).cuda()
    # model = torch.load(path_model)

    print(count_params(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=epochs, div_factor=2,
                                                final_div_factor=1e-3/1e-6, pct_start=0.2)
    MSELoss = nn.MSELoss(reduction='mean').cuda()
    p0 = torch.zeros((batch_size, T - 2, S, S)).cuda()
    for ep in range(epochs):
        t1 = default_timer()
        model.train()
        train_mse = 0
        train_lp = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x)
            mse = MSELoss(out[:, train_grid], y)
            pi = derivative(out)
            lp = MSELoss(pi, p0)
            loss = mse + 1e0 * lp
            loss.backward()

            optimizer.step()
            train_mse += mse.item() / ntrain
            train_lp += lp.item() / ntrain
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
                    out = model(x)
                    loss = MSELoss(out, y)
                    test_loss += loss.item() / ntest
                t2 = default_timer()
                t_test = t2 - t1
            print(f"{ep:3d}, Test_loss: {test_loss:.4e}, time test: {t_test:2.3f}")
            torch.save(model, path_model)
            print('Model saving: ', path_model)
