import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.imageencoder import ImageEncoder
from networks.imagedecoder import ImageDecoder

class ConvVae(nn.Module):
    def __init__(self, img_res=32, z_dim=16, layers_channels=[4,8,16]):
        super(ConvVae, self).__init__()

        self.img_res = img_res
        self.z_dim = z_dim
        self.img_dim = img_res * img_res

        self.layers_channels = layers_channels
        self.div = math.pow(2, len(self.layers_channels))
        self.fc_input_size = self.layers_channels[-1] * int(self.img_res / self.div) * int(self.img_res / self.div)
        in_channels = 1
        self.encoder = ImageEncoder(in_channels, self.layers_channels, 'enc')
        self.fc11 = nn.Linear(self.fc_input_size, z_dim)
        self.fc12 = nn.Linear(self.fc_input_size, z_dim)
        self.fc2 = nn.Linear(z_dim, self.fc_input_size)
        self.decoder = ImageDecoder(list(reversed(self.layers_channels)), in_channels, 'dec')

    def encode(self, data_batch):
        ip_enc = torch.reshape(data_batch, [-1, 1, self.img_res, self.img_res])
        z = self.encoder(ip_enc).view(-1, self.fc_input_size)
        return self.fc11(z), self.fc12(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc2(z)).view(-1, self.layers_channels[-1], int(self.img_res / self.div), int(self.img_res / self.div))
        op = self.decoder(h3)
        return op

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar