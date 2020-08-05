import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from networks.reshape import Reshape

class ImageDecoder(nn.Module):
    def __init__(self, layers_channels, output_channels, prefix, addFlatten=True, useSigmoid=True, kernel_size=3):
        super(ImageDecoder, self).__init__()

        layers = OrderedDict()
        padding = kernel_size // 2
        for i in range(len(layers_channels)-1):
            layers[prefix + '_convt' + str(i)] = nn.ConvTranspose2d(in_channels=layers_channels[i],
                                        out_channels=layers_channels[i+1],
                                        kernel_size=kernel_size, stride=2, padding=padding, output_padding=1)
            layers[prefix + '_relu' + str(i)] = nn.ReLU()

        last = len(layers_channels)-1
        layers[prefix + '_convt' + str(last)] = nn.ConvTranspose2d(in_channels=layers_channels[last],
                                    out_channels=output_channels,
                                    kernel_size=kernel_size, stride=2, padding=padding, output_padding=1)
        if useSigmoid:
            layers[prefix + '_sigmoid' + str(last)] = nn.Sigmoid()
        if addFlatten:
            layers[prefix + '_flat' + str(last)] = nn.Flatten()

        self.net = nn.Sequential(layers)

    def forward(self, data):
        return self.net(data)

class ImageDecoderFlatInput(ImageDecoder):
    def __init__(self, z_dim, layers_channels, output_channels, prefix, addFlatten=True, useSigmoid=True):
        super(ImageDecoderFlatInput, self).__init__(layers_channels, output_channels, prefix, addFlatten, useSigmoid)
        self.z_dim = z_dim
        img_dim = int(np.sqrt((z_dim / layers_channels[0])))
        self.reshapeInput = Reshape(-1, layers_channels[0], img_dim, img_dim)

    def forward(self, data):
        return self.net(self.reshapeInput(data))

