import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from networks.reshape import Reshape

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