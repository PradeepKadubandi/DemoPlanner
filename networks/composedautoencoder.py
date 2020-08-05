import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from networks.reshape import Reshape
from networks.imageencoder import ImageEncoder
from networks.imagedecoder import ImageDecoder

class ComposedAutoEncoder(nn.Module):
    def __init__(self, x_dim=2, img_res=32, z_dim=2, 
                useMaxPool=False, layers_channels = [4, 8, 16],
                useSigmoid=True, device=None, kernel_size=3):
        super(ComposedAutoEncoder, self).__init__()
        self.x_dim = x_dim
        self.img_res = img_res
        self.z_dim = z_dim
        self.img_dim = self.img_res * self.img_res
        self.device = device
        in_channels = 1
        self.encoder = ImageEncoder(in_channels, layers_channels, 'enc', useMaxPool=useMaxPool, kernel_size=kernel_size)
        self.decoder = ImageDecoder(list(reversed(layers_channels)), in_channels, 'dec', useSigmoid=useSigmoid, kernel_size=kernel_size)

    def forward(self, data_batch):
        ip_enc = torch.reshape(data_batch, [-1, 1, self.img_res, self.img_res])
        ip_enc = ip_enc.to(self.device)
        z = self.encoder(ip_enc)
        op = self.decoder(z)
        return op