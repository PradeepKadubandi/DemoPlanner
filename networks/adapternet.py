import torch.nn as nn

class AdapterNet(nn.Module):
    def __init__(self, testNet, ip_adapter=None, op_adapter=None):
        super().__init__()
        self.testNet = testNet
        self.ip_adapter = ip_adapter if ip_adapter else lambda x:x
        self.op_adapter = op_adapter if op_adapter else lambda x:x
        
    def forward(self, data):
        ip = self.ip_adapter(data)
        op = self.testNet(ip)
        return self.op_adapter(op)