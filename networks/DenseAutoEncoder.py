import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from networks.reshape import Reshape

class DenseAutoEncoder(nn.Module):
    def __init__(self, enc_layer_dims):
        super(DenseAutoEncoder, self).__init__()

        layers = OrderedDict()
        for i in range(len(enc_layer_dims)-1):
            layers['enc_fc' + str(i)] = nn.Linear(enc_layer_dims[i], enc_layer_dims[i+1])
            layers['enc_relu' + str(i)] = nn.ReLU()
        self.encoder = nn.Sequential(layers)

        layers = OrderedDict()
        for i in range(len(enc_layer_dims)-1, 1, -1):
            layers['dec_fc' + str(i)] = nn.Linear(enc_layer_dims[i], enc_layer_dims[i-1])
            layers['dec_relu' + str(i)] = nn.ReLU()
        layers['dec_fc1'] = nn.Linear(enc_layer_dims[1], enc_layer_dims[0])
        layers['dec_sig1'] = nn.Sigmoid()
        self.decoder = nn.Sequential(layers)

    def forward(self, data_batch):
        z = self.encoder(data_batch)
        op = self.decoder(z)
        return op