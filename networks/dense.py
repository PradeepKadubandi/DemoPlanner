import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from networks.reshape import Reshape

class Dense(nn.Module):
    def __init__(self, layer_dims, bias=True):
        super(Dense, self).__init__()

        layers = OrderedDict()
        for i in range(len(layer_dims)-2):
            layers['enc_fc' + str(i)] = nn.Linear(layer_dims[i], layer_dims[i+1], bias=bias)
            layers['enc_relu' + str(i)] = nn.ReLU()
        i = len(layer_dims)-2
        layers['enc_fc' + str(i)] = nn.Linear(layer_dims[i], layer_dims[i+1], bias=bias)
        layers['enc_sig' + str(i)] = nn.Sigmoid()
        self.net = nn.Sequential(layers)

    def forward(self, data_batch):
        op = self.net(data_batch)
        return op