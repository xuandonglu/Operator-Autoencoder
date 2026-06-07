import torch
from torch import nn
import torch.nn.functional as F

max_channel = 256
init_dim = 1
class DPN(nn.Module):
    def __init__(self, input_dim, latent_dim, pattern):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = self.buildEncoder(latent_dim)
        self.decoder = self.buildDecoder(latent_dim)

    def buildEncoder(self, latent_dim):
        m = 1 << (latent_dim - 1).bit_length()
        encoder = nn.Sequential(ResidualBlock(self.input_dim[0]*self.input_dim[1], m, m, 8),
                               ResidualBlock(m, m, latent_dim, 8),)
        return encoder

    def buildDecoder(self, latent_dim):
        m = 1 << (latent_dim - 1).bit_length()
        decoder = nn.Sequential(ResidualBlock(latent_dim, m, m, 8),
                               ResidualBlock(m, m, self.input_dim[0]*self.input_dim[1], 8),)
        return decoder

    def forward(self, data):
        z = self.encode(data)
        reconstruction = self.decode(z)
        return reconstruction

    def encode(self, data):
        z = self.encoder(data.reshape([data.shape[0], data.shape[-1]*data.shape[-2]]))
        return z

    def decode(self, z):
        rec = self.decoder(z)
        return rec.reshape([rec.shape[0], 1, self.input_dim[0], self.input_dim[1]])

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