import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class autoencoder(torch.torch.nn.Module):
    def __init__(self, device=None):
        super(autoencoder, self).__init__()
        self.device = device
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1), #bias=False
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        x = torch.reshape(x, [-1, 1, 32, 32])
        x = x.to(self.device)
        x = self.encoder(x)
        x = self.decoder(x)
        return x