import torch
import torch.nn as nn
from collections import OrderedDict

class ImageDecoder(nn.Module):
    def __init__(self, layers_channels, output_channels, prefix, addFlatten=True):
        super(ImageDecoder, self).__init__()

        layers = OrderedDict()
        for i in range(len(layers_channels)-1):
            layers[prefix + '_convt' + str(i)] = nn.ConvTranspose2d(in_channels=layers_channels[i],
                                        out_channels=layers_channels[i+1],
                                        kernel_size=3, stride=2, padding=1, output_padding=1)
            layers[prefix + '_relu' + str(i)] = nn.ReLU()

        last = len(layers_channels)-1
        layers[prefix + '_convt' + str(last)] = nn.ConvTranspose2d(in_channels=layers_channels[last],
                                    out_channels=output_channels,
                                    kernel_size=3, stride=2, padding=1, output_padding=1)
        layers[prefix + '_sigmoid' + str(last)] = nn.Sigmoid()
        if addFlatten:
            layers[prefix + '_flat' + str(last)] = nn.Flatten()

        self.net = nn.Sequential(layers)

    def forward(self, data):
        return self.net(data)

