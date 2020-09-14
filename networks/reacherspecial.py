import torch
import torch.nn as nn
import simplereacheradapters

class LatentPolicyNet(nn.Module):
    def __init__(self, latent, policy):
        super(LatentPolicyNet, self).__init__()
        self.latent = latent
        self.policy = policy

    def forward(self, data):
        policy_input = self.latent(data)
        policy_output = self.policy(policy_input)
        return policy_output

class LatentPolicyWithGroundTruthXNet(nn.Module):
    '''
    Input : (I,x)
    Output : u
    latent is a sub network that takes I and returns y'
    policy takes (x, y') and returns u
    '''
    def __init__(self, latent, policy):
        super(LatentPolicyWithGroundTruthXNet, self).__init__()
        self.latent = latent
        self.policy = policy

    def forward(self, data):
        y_hat = self.latent(simplereacheradapters.It_from_XtIt_adapter(data))
        policy_input = torch.cat((simplereacheradapters.Xt_from_XtIt_adapter(data), y_hat), dim=1)
        policy_output = self.policy(policy_input)
        return policy_output