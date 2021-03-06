import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters import *
from networks.ArgMax import ArgMaxFunction
from networks.imageencoder import ImageEncoderFlatInput
from networks.imagedecoder import ImageDecoderFlatInput
from networks.dense import Dense
from datagen.discreteenvironment import DiscreteEnvironment

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

class ImageToEnvAndMultiNet(nn.Module):
    def __init__(self, ItoY, policy, dynamics):
        super(ImageToEnvAndMultiNet, self).__init__()
        self.ItoY = ItoY
        self.policy = policy
        self.dynamics = dynamics

    def forward(self, data):
        env = self.ItoY(It_scaled_adapter(data))
        xt = Xt_scaled_adapter(data)
        policy_input = torch.cat((xt, env[:, 2:]), dim=1) # Use prediction of only y from image to env network, use original x
        policy_output = self.policy(policy_input)
        lsmax_x = F.log_softmax(policy_output[:, :3], dim=1)
        lsmax_y = F.log_softmax(policy_output[:, 3:], dim=1)
        ux = ArgMaxFunction.apply(lsmax_x)
        uy = ArgMaxFunction.apply(lsmax_y)
        ut = (torch.cat((ux,uy), dim=1)) / 2.0
        dynamics_input = torch.cat((xt,ut), dim=1)
        return self.dynamics(dynamics_input)

    def rollout(self, gt_trajectory):
        de = DiscreteEnvironment()
        Ihat_tplus = It_scaled_adapter(gt_trajectory[:1, :])
        xhat_tplus = Xt_unscaled_adapter(gt_trajectory[:1, :])
        predictions = torch.zeros(len(gt_trajectory), 1030) # (yhat_t, uhat_t, xhat_t+1, Ihat_t+1) = 1030
        for i in range(len(gt_trajectory)):
            Yhat_t = self.ItoY(Ihat_tplus.view((-1, 1, 32, 32)))
            yhat_t = Yhat_t[:, 2:]
            xt = Xt_scaled_adapter(gt_trajectory[i:i+1, :])
            policy_input = torch.cat((xt, yhat_t), dim=1) # Use prediction of only y from image to env network, use original x
            policy_output = self.policy(policy_input)
            lsmax_x = F.log_softmax(policy_output[:, :3], dim=1)
            lsmax_y = F.log_softmax(policy_output[:, 3:], dim=1)
            ux = ArgMaxFunction.apply(lsmax_x)
            uy = ArgMaxFunction.apply(lsmax_y)
            uhat_t = (torch.cat((ux,uy), dim=1)) / 2.0
            dynamics_input = torch.cat((xt,uhat_t), dim=1)
            dynamics_out = self.dynamics(dynamics_input)
            xhat_tplus += ((dynamics_out * 2.0) - 1.0)  # Dynamics network predicts x_t+1 - x_t but in the range (0, 1)
            start = xhat_tplus[0].int() # Upscale the output from dynamics to image size
            goal = (32 * yhat_t[0]).int()
            image = torch.Tensor(de.generateImage(start, goal)) # generate image
            predictions[i, :] = torch.cat((yhat_t[0] * 32, (uhat_t[0] * 2.0) - 1.0, xhat_tplus[0], image.view(-1)))
            Ihat_tplus = image / 255.0 # scale the prediction back to expected range for network
        return predictions

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
