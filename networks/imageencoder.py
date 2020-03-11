import torch
import torch.nn as nn
from collections import OrderedDict

class ImageEncoder(nn.Module):
    def __init__(self, input_channels, layers_channels, prefix):
        super(ImageEncoder, self).__init__()

        layers = OrderedDict()
        pr_ch = input_channels
        for i in range(len(layers_channels)):
            layers[prefix + '_conv' + str(i)] = nn.Conv2d(in_channels=pr_ch,
                                        out_channels=layers_channels[i],
                                        kernel_size=3, stride=2, padding=1)
            layers[prefix + '_relu' + str(i)] = nn.ReLU()
            pr_ch = layers_channels[i]
        self.net = nn.Sequential(layers)

    def forward(self, data):
        return self.net(data)

