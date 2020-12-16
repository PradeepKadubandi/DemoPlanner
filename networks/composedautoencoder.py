import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from networks.reshape import Reshape
from networks.imageencoder import ImageEncoder
from networks.imagedecoder import ImageDecoder
from networks.dense import Dense
import math

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

class CnnFcAutoEncoder(nn.Module):
    def __init__(self, cnn_layer_channels, fc_layer_channels,
                x_dim=3, img_res=64, z_dim=256):
        super(CnnFcAutoEncoder, self).__init__()
        self.x_dim = x_dim
        self.img_res = img_res
        self.z_dim = z_dim
        self.cnn_layer_channels = cnn_layer_channels
        self.fc_layer_channels = fc_layer_channels
        in_channels = 1
        self.img_encoder = ImageEncoder(in_channels, cnn_layer_channels, 'img_enc', useMaxPool=True, kernel_size=3)
        self.dense_encoder = Dense(fc_layer_channels, last_act='relu')
        self.dense_decoder = Dense(list(reversed(fc_layer_channels)), last_act='relu')
        self.img_decoder = ImageDecoder(list(reversed(cnn_layer_channels)), in_channels, 'img_dec', useSigmoid=True, kernel_size=3)

    def forward(self, data_batch):
        x = data_batch.view((-1, 1, self.img_res, self.img_res))
        x = self.img_encoder(x)
        x = x.view((-1, x.size()[1:].numel()))
        x = self.dense_encoder(x)
        x = self.dense_decoder(x)
        last_layer_channels = self.cnn_layer_channels[-1]
        mid_img_dim = x.size()[1:].numel() // last_layer_channels
        mid_img_res = int(math.sqrt(mid_img_dim))
        x = x.view((-1, last_layer_channels, mid_img_res, mid_img_res))
        x = self.img_decoder(x)
        return x