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