import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
class DPN(nn.Module):
    def __init__(self, input_dim, latent_dim, pattern):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim_f = latent_dim
        self.latent_dim_m = int(latent_dim / 2)
        self.latent_dim_c = int(latent_dim / 4)
        self.branch_net = self.buildBranch(self.latent_dim_f + self.latent_dim_m + self.latent_dim_c, pattern)
        self.trunk_net_f = self.buildTrunk(self.latent_dim_f, pattern)
        self.trunk_net_m = self.buildTrunk(self.latent_dim_m, pattern)
        self.trunk_net_c = self.buildTrunk(self.latent_dim_c, pattern)

    def buildBranch(self, latent_dim, pattern):
        m = 1 << (latent_dim - 1).bit_length()
        branch = nn.Sequential(ResidualBlock(self.input_dim[0]*self.input_dim[1], m, m, 8),
                               ResidualBlock(m, m, latent_dim, 8),)

        if pattern == 'fno' or pattern == 'fno1':
            model = nn.Sequential(FNO1(24, 24, 1, 1), branch)
        else:
            model = branch
        return model

    def buildTrunk(self, latent_dim, pattern):
        trunk = nn.Sequential(ResidualBlock(2, 128, 128, 8),
                              ResidualBlock(128, 128, latent_dim, 8),)
        if pattern == 'fno' or pattern == 'fno2':
            model = FNO2(16, 16, latent_dim)
        else:
            model = trunk
        return model

    def forward(self, data):
        z = self.encode(data)
        zf, zm, zc = torch.split(z,[self.latent_dim_f, self.latent_dim_m, self.latent_dim_c], dim=-1)
        rec = self.decode(zf, zm, zc)
        return rec

    def encode(self, data):
        x = data.reshape([data.shape[0], data.shape[-1]*data.shape[-2]])
        branch_out = self.branch_net(x)
        return branch_out

    def trunk(self, device):
        gf = self.get_grid(self.input_dim[0], self.input_dim[1], device)
        trunk_out_f = self.trunk_net_f(gf)
        gm = self.get_grid(int(self.input_dim[0] / 2), int(self.input_dim[1] / 2), device)
        trunk_out_m = self.trunk_net_m(gm)
        gc = self.get_grid(int(self.input_dim[0] / 4), int(self.input_dim[1] / 4), device)
        trunk_out_c = self.trunk_net_c(gc)
        return trunk_out_f, trunk_out_m, trunk_out_c

    def decode(self, zf, zm, zc):
        trunk_out_f, trunk_out_m, trunk_out_c = self.trunk(zc.device)
        yf = torch.einsum("bi,xyi->bxy", zf, trunk_out_f)
        ym = torch.einsum("bi,xyi->bxy", zm, trunk_out_m)
        yc = torch.einsum("bi,xyi->bxy", zc, trunk_out_c)
        ym = F.interpolate(ym.unsqueeze(1), size=yf.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        yc = F.interpolate(yc.unsqueeze(1), size=yf.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        y = yf + ym + yc
        return y

    def get_grid(self, size_x, size_y, device):
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(size_x, 1, 1).repeat([1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, size_y, 1).repeat([size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO1(nn.Module):
    def __init__(self, modes1, modes2, width, input_dim):
        super(FNO1, self).__init__()
        self.p = nn.Linear(input_dim, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.q = nn.Linear(width, input_dim)

    def forward(self, x):
        x = self.p(x.reshape([x.shape[0], 768, 512, 1]))
        x = x.permute(0, 3, 1, 2)
        x = self.conv0(x)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.q(x)
        return x.reshape([x.shape[0], x.shape[1]*x.shape[2]])

class FNO2(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.conv0 = SpectralConv2d(50, 50, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(50, 50, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(50, 50, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(50, 50, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(50, 50, 1)
        self.w1 = nn.Conv2d(50, 50, 1)
        self.w2 = nn.Conv2d(50, 50, 1)
        self.w3 = nn.Conv2d(50, 50, 1)
        self.fc0 = nn.Linear(2, 50)
        self.fc1 = nn.Linear(50, width)

    def forward(self, x):
        x = self.fc0(x.unsqueeze(0))
        x = x.permute(0, 3, 1, 2)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        return x.squeeze(0)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.residual(x)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
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
