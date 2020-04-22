from networks.imageencoder import *
from networks.imagedecoder import *
from networks.dense import Dense
from adapters import *

'''
Contains networks that were built for one-off trials and specialized
purpose.
'''

class PreTrainedConvWithNewFc(nn.Module):
    '''
    Use the pre-trained image auto encoder (ComposedAutoEncoder type)
    and introduce FC layer to reduce hidden dimension.
    '''
    def __init__(self, cae):
        super(PreTrainedConvWithNewFc, self).__init__()
        self.cae = cae
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 64)
        
    def forward(self, data):
        op = self.cae.encoder(data.view(-1, 1, 32, 32))
        op = op.view(op.size()[0], -1)
        op = self.fc2(self.fc1(op))
        op = op.view(op.size()[0], 16, 2, 2)
        op = self.cae.decoder(op)
        return op

class ImageEnvEncoder(nn.Module):
    '''
    Uses image encoder + env encoder and training should be done on loss between 2 hidden dims.
    Uses also image decoder and env decoder to include reconstruction loss.
    Expects the identity adapter as input adapter.
    '''
    def __init__(self, imgEnc=None, envEnc=None, imgDec=None, envDec=None):
        super(ImageEnvEncoder, self).__init__()
        self.imgEnc = imgEnc if imgEnc else ImageEncoderFlatInput(1, [16,16,16,16], useMaxPool=True, addFlatten=True, prefix='imgenc')
        self.envEnc = envEnc if envEnc else Dense([4, 16, 64], use_last_act=False, prefix='envenc')
        self.imgDec = imgDec if imgDec else ImageDecoderFlatInput(64, [16,16,16,16], output_channels=1, prefix='imgdec')
        self.envDec = envDec if envDec else Dense([64, 16, 4], last_act='sigmoid', prefix='EnvDec')

    def forward(self, data):
        z_img = self.imgEnc(It_scaled_adapter(data))
        z_env = self.envEnc(XtYt_scaled_adapter(data))
        I_recon = self.imgDec(z_img)
        y_recon = self.envDec(z_env)
        return z_img, z_env, I_recon, y_recon