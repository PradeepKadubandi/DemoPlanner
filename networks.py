import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

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

class ComposedAutoEncoder(nn.Module):
    def __init__(self, x_dim=2, img_res=32, z_dim=2):
        super(ComposedAutoEncoder, self).__init__()
        self.x_dim = x_dim
        self.img_res = img_res
        self.z_dim = z_dim
        self.img_dim = self.img_res * self.img_res
        self.encoder = ImageEncoder(1, [4, 8, 16], 'enc')
        self.decoder = ImageDecoder([16, 8, 4], 1, 'dec')

    def forward(self, data_batch):
        ip_enc = torch.reshape(data_batch, [-1, 1, self.img_res, self.img_res])
        return self.decoder(self.encoder(ip_enc))

class AutoEncoder(nn.Module):
    def __init__(self, x_dim=2, img_res=32, z_dim=2):
        super(AutoEncoder, self).__init__()

        self.x_dim = x_dim
        self.img_res = img_res
        self.z_dim = z_dim
        self.img_dim = self.img_res * self.img_res

        in_ch = 1
        l1_ch = 4
        l2_ch = 8
        l3_ch = 16

        self.encoder = nn.Sequential(OrderedDict([
            ('enc_conv1', nn.Conv2d(in_channels=in_ch, out_channels=l1_ch, kernel_size=3, stride=2, padding=1)),
            ('enc_relu1', nn.ReLU()),
            ('enc_conv2', nn.Conv2d(l1_ch, l2_ch, 3, stride=2, padding=1)),
            ('enc_relu2', nn.ReLU()),
            ('enc_conv3', nn.Conv2d(l2_ch, l3_ch, 3, stride=2, padding=1)),
            ('enc_relu3', nn.ReLU()),
            # ('enc_reshape1', Reshape(-1, 256)),
            # ('enc_fc1', nn.Linear(256, 4)),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            # ('dec_fc1', nn.Linear(4, 256)),
            # ('dec_relu4', nn.ReLU()),
            # ('dec_reshape1', Reshape(-1, l3_ch, 4, 4)),
            ('dec_convt3', nn.ConvTranspose2d(in_channels=l3_ch, out_channels=l2_ch, kernel_size=3, stride=2, padding=1, output_padding=1)),
            ('dec_relu3', nn.ReLU()),
            ('dec_convt2', nn.ConvTranspose2d(l2_ch, l1_ch, 3, stride=2, padding=1, output_padding=1)),
            ('dec_relu2', nn.ReLU()),
            ('dec_conv1', nn.ConvTranspose2d(l1_ch, in_ch, 3, stride=2, padding=1, output_padding=1)),
            ('dec_sigmoid1', nn.Sigmoid()),
            ('dec_flat1', nn.Flatten()),
        ]))

    def forward(self, data_batch):
        ip_enc = torch.reshape(data_batch, [-1, 1, self.img_res, self.img_res])
        z = self.encoder(ip_enc)
        output = self.decoder(z)
        return output