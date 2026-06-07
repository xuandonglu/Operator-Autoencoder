import torch
from torch import nn
import torch.nn.functional as F

max_channel = 256
init_dim = 1
class AE(nn.Module):
    def __init__(self, latent_dim, pattern, mode, width):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = self.buildEncoder(latent_dim)
        self.decoder = self.buildDecoder(latent_dim, pattern, mode, width)

    def buildEncoder(self, latent_dim):
        encoder = nn.Sequential(
            nn.Conv1d(in_channels=init_dim, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(max_channel * 8, max_channel),
            nn.ELU(),
            nn.Linear(max_channel, latent_dim),
        )
        return encoder

    def buildDecoder(self, latent_dim, pattern, mode, width):
        decoder = nn.Sequential(
            nn.Linear(latent_dim, max_channel),
            nn.ELU(),
            nn.Linear(max_channel, max_channel * 8),
            nn.ELU(),
            nn.Unflatten(dim=1, unflattened_size=(max_channel, 8)),
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(in_channels=8, out_channels=init_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        if pattern == 'fno':
            model = nn.Sequential(decoder, FNO(mode, width, 1))
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

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO(nn.Module):
    def __init__(self, modes1, width, input_dim):
        super(FNO, self).__init__()
        self.p = nn.Linear(input_dim, width)  # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv1d(width, width, modes1)
        self.q = nn.Linear(width, input_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.p(x)
        x = x.permute(0, 2, 1)
        x = self.conv0(x)
        x = F.gelu(x)
        x = x.permute(0, 2, 1)
        x = self.q(x)
        x = x.permute(0, 2, 1)
        return x