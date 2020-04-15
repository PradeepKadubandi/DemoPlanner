import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from networks.reshape import Reshape

class Dense(nn.Module):
    def __init__(self, layer_dims, bias=True, last_act='sigmoid', use_last_act=True, prefix='den'):
        super(Dense, self).__init__()

        layers = OrderedDict()
        for i in range(len(layer_dims)-2):
            layers[prefix + '_fc' + str(i)] = nn.Linear(layer_dims[i], layer_dims[i+1], bias=bias)
            layers[prefix + '_relu' + str(i)] = nn.ReLU()
        i = len(layer_dims)-2
        layers[prefix + '_fc' + str(i)] = nn.Linear(layer_dims[i], layer_dims[i+1], bias=bias)
        if use_last_act:
            if last_act == 'sigmoid':
                layers[prefix + '_sig' + str(i)] = nn.Sigmoid()
            elif last_act == 'tanh':
                layers[prefix + '_tanh' + str(i)] = nn.Tanh()
            elif last_act == 'relu':
                layers[prefix + '_relu' + str(i)] = nn.ReLU()
            else:
                raise Exception('Unexpected value for last_act parameter')
        # layers['enc_softmax'] = nn.Softmax()
        self.net = nn.Sequential(layers)

    def forward(self, data_batch):
        op = self.net(data_batch)
        return op

class DenseForPolicy(nn.Module):
    def __init__(self, layer_dims):
        super(DenseForPolicy, self).__init__()
        self.dense = Dense(layer_dims[:-1], use_last_act=False)
        self.x = nn.Linear(layer_dims[-2], layer_dims[-1])
        self.y = nn.Linear(layer_dims[-2], layer_dims[-1])

    def forward(self, data_batch):
        op = self.dense(data_batch)
        op_x = F.softmax(self.x(op))
        op_y = F.softmax(self.y(op))
        return torch.cat((op_x, op_y), axis=1)

