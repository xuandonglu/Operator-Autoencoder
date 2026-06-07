import torch
from torch import nn
import torch.nn.functional as F

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim, mode, width):
        super().__init__()
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = self.buildEncoder(latent_dim)
        self.decoder = self.buildDecoder(latent_dim)

    def buildEncoder(self, latent_dim):
        m = 1 << (latent_dim - 1).bit_length()
        input_dim = self.input_dim
        return nn.Sequential(
            ResidualBlock(input_dim, 64, 64, 8),
            ResidualBlock(64, m, latent_dim, 8),
        )

    def buildDecoder(self, latent_dim):
        m = 1 << (latent_dim - 1).bit_length()
        output_dim = self.input_dim
        model = nn.Sequential(
            ResidualBlock(latent_dim, m, m, 8),
            ResidualBlock(m, m, output_dim, 8),
        )
        return model

    def forward(self, data):
        z = self.encode(data)
        return self.decode(z)

    def encode(self, data):
        return self.encoder(data.reshape([data.shape[0], -1]))

    def decode(self, z):
        rec = self.decoder(z)
        return rec.reshape([rec.shape[0], self.input_dim])


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.residual(x)

