import torch
from torch import nn
import torch.nn.functional as F

max_channel = 128
init_dim = 1
class AE(nn.Module):
    def __init__(self, latent_dim, pattern, mode, width):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = self.buildEncoder(latent_dim)
        self.decoder = self.buildDecoder(latent_dim, pattern, mode, width)

    def buildEncoder(self, latent_dim):
        encoder = nn.Sequential(
            nn.Conv2d(in_channels=init_dim, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(max_channel * 4, max_channel),
            nn.ELU(),
            nn.Linear(max_channel, latent_dim),
        )
        return encoder

    def buildDecoder(self, latent_dim, pattern, mode, width):
        decoder = nn.Sequential(
            nn.Linear(latent_dim, max_channel),
            nn.ELU(),
            nn.Linear(max_channel, max_channel * 4),
            nn.ELU(),
            nn.Unflatten(dim=1, unflattened_size=(max_channel, 2, 2)),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=init_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        if pattern == 'fno':
            model = nn.Sequential(decoder, FNO(mode, mode, width, 1))
        else:
            model = decoder
        return model

    def forward(self, data):
        z = self.encoder(data)
        reconstruction = self.decoder(z)
        return reconstruction

    def encode(self, data):
        z = self.encoder(data)
        return z

    def decode(self, data):
        return self.decoder(data)

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

class FNO(nn.Module):
    def __init__(self, modes1, modes2, width, input_dim):
        super(FNO, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.fc0 = nn.Linear(input_dim, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, input_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x.squeeze(-1)
