import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters import *
from networks.ArgMax import ArgMaxFunction
from networks.imageencoder import ImageEncoderFlatInput
from networks.imagedecoder import ImageDecoderFlatInput
from networks.dense import Dense

class MultiNet(nn.Module):
    '''
    This is a combination net to train policy and dynamics network together.
    '''
    def __init__(self, policy, dynamics):
        super(MultiNet, self).__init__()
        self.policy = policy
        self.dynamics = dynamics
        
    def forward(self, data):
        policy_input = policy_input_adapter(data)
        policy_output = self.policy(policy_input)
        lsmax_x = F.log_softmax(policy_output[:, :3], dim=1)
        lsmax_y = F.log_softmax(policy_output[:, 3:], dim=1)
        ux = ArgMaxFunction.apply(lsmax_x)
        uy = ArgMaxFunction.apply(lsmax_y)
        ut = (torch.cat((ux,uy), dim=1)) / 2.0
        xt = Xt_scaled_adapter(data)
        dynamics_input = torch.cat((xt,ut), dim=1)
        return self.dynamics(dynamics_input)

class MultiEncoderNet(nn.Module):
    '''
    An image encoder, image decoder, env encoder, env decoder all in one net.
    Can train different combinations by setting the right optimizers.
    Uses hardcoded best known configurations for network structures, will
    build flexibility into network structure later.
    '''
    def __init__(self):
        super(MultiEncoderNet, self).__init__()
        self.z_dim = 64
        imgLayers = [16,16,16,16]
        envLayers = [4,16,64]
        self.imgenc = ImageEncoderFlatInput(input_channels=1, layers_channels=imgLayers, prefix='ImgEnc', addFlatten=True, useMaxPool=True)
        self.imgdec = ImageDecoderFlatInput(self.z_dim, list(reversed(imgLayers)), output_channels=1, prefix='ImgDec')
        self.envenc = Dense(envLayers, use_last_act=False, prefix='EnvEnc')
        self.envdec = Dense(list(reversed(envLayers)), last_act='sigmoid', prefix='EnvDec')
        # self.common = nn.Linear(self.z_dim, self.z_dim)
        # self.setNetworksForForward('I', 'I')

    def __set_requires_grad(self, net, val):
        for p in net.parameters():
            p.requires_grad_(val)

    def setNetworksForForward(self, encStr, decStr):
        for net in [self.imgdec, self.imgenc, self.envdec, self.envenc]:
            self.__set_requires_grad(net, False)
        self.encoder = self.imgenc if encStr == 'I' else self.envenc
        self.decoder = self.imgdec if decStr == 'I' else self.envdec
        for net in [self.encoder, self.decoder]:
            self.__set_requires_grad(net, True)

    def forward(self, data):
        It = It_scaled_adapter(data)
        Yt = XtYt_scaled_adapter(data)
        I_to_I = self.imgdec(self.imgenc(It))
        I_to_Y = self.envdec(self.imgenc(It))
        Y_to_I = self.imgdec(self.envenc(Yt))
        Y_to_Y = self.envdec(self.envenc(Yt))
        return I_to_I, I_to_Y, Y_to_I, Y_to_Y
