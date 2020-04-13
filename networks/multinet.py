import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters import *
from networks.ArgMax import ArgMaxFunction

class MultiNet(nn.Module):
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