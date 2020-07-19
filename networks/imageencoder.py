import torch
import torch.nn as nn
from collections import OrderedDict
from networks.reshape import Reshape

class ImageEncoder(nn.Module):
    def __init__(self, input_channels, layers_channels, prefix, useMaxPool=False, addFlatten=False):
        '''
        If useMaxPool is set to True, Max pooling is used to reduce
        the image dims instead of stride = 2.
        '''
        super(ImageEncoder, self).__init__()

        layers = OrderedDict()
        pr_ch = input_channels
        stride = 1 if useMaxPool else 2
        for i in range(len(layers_channels)):
            layers[prefix + '_conv' + str(i)] = nn.Conv2d(in_channels=pr_ch,
                                        out_channels=layers_channels[i],
                                        kernel_size=3, stride=stride, padding=1)
            layers[prefix + '_relu' + str(i)] = nn.ReLU()
            if (useMaxPool):
                layers[prefix + '_maxpool' + str(i)] = nn.MaxPool2d(2, stride=2)
            pr_ch = layers_channels[i]
        if addFlatten:
            layers[prefix + '_flat'] = nn.Flatten()
        self.net = nn.Sequential(layers)

    def forward(self, data):
        return self.net(data)

class ImageEncoderFlatInput(ImageEncoder):
    def __init__(self, input_channels, layers_channels, prefix, useMaxPool=False, addFlatten=False, img_res=32):
        super(ImageEncoderFlatInput, self).__init__(input_channels, layers_channels, prefix, useMaxPool, addFlatten)
        self.reshapeInput = Reshape(-1, input_channels, img_res, img_res)

    def forward(self, data):
        return self.net(self.reshapeInput(data))

