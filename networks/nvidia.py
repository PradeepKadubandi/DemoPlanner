'''
Implemention of network from paper : https://arxiv.org/pdf/1604.07316.pdf
End to End learning for self-driving cars by Nvidia
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Nvidia(nn.Module):
    def __init__(self, in_channels=3, conv_channels=[24, 36, 48, 64, 64], fc_layers=[64, 32, 16, 3], last_act='tanh'):
        '''
        The default parameters are all closely matching the paper but adjusted to fit
        the input size of 64 x 64 image. For a different input size, different channels must be used.
        '''
        super(Nvidia, self).__init__()
        layers = (OrderedDict([
            ('conv1', nn.Conv2d(in_channels=in_channels, out_channels=conv_channels[0], kernel_size=5, stride=2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(conv_channels[0], conv_channels[1], 5, stride=2)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(conv_channels[1], conv_channels[2], 5, stride=2)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(conv_channels[2], conv_channels[3], 3, stride=1)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv2d(conv_channels[3], conv_channels[4], 3, stride=1)),
            ('relu5', nn.ReLU()),
            ('flat5', nn.Flatten()),
            ('fc6', nn.Linear(fc_layers[0], fc_layers[1])),
            ('relu6', nn.ReLU()),
            ('fc7', nn.Linear(fc_layers[1], fc_layers[2])),
            ('relu7', nn.ReLU()),
            ('fc8', nn.Linear(fc_layers[2], fc_layers[3]))
        ]))
        if last_act == 'sigmoid':
            layers['sig8'] = nn.Sigmoid()
        elif last_act == 'tanh':
            layers['tanh8'] = nn.Tanh()
        elif last_act == 'relu':
            layers['relu8'] = nn.ReLU()
        else:
            raise Exception('Unexpected value for last_act parameter')
        self.net = nn.Sequential(layers)

    def forward(self, data):
        return self.net(data)