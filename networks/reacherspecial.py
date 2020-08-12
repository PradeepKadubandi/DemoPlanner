class LatentPolicyNet(nn.Module):
    def __init__(self, latent, policy, latentFilter=None):
        '''
        latentFilter : (inputDataRow, latentOutput) --> policyInput
        '''
        super(LatentPolicyNet, self).__init__()
        self.latent = latent
        self.policy = policy
        self.latentFilter = latentFilter

    def forward(self, data):
        policy_input = self.latent(data)
        if self.latentFilter:
            policy_input = self.latentFilter(data, policy_input)
        policy_output = self.policy(policy_input)
        return policy_output

