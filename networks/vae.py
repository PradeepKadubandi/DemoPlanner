import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, img_res=32, z_dim=16):
        super(VAE, self).__init__()

        self.img_res = img_res
        self.z_dim = z_dim
        self.img_dim = img_res * img_res

        hl1 = int(self.img_dim / 2)
        hl2 = int(hl1 / 2)

        self.fc1 = nn.Linear(self.img_dim, hl1)
        self.fc2 = nn.Linear(hl1, hl2)
        self.fc31 = nn.Linear(hl2, self.z_dim)
        self.fc32 = nn.Linear(hl2, self.z_dim)
        self.fc4 = nn.Linear(self.z_dim, hl2)
        self.fc5 = nn.Linear(hl2, hl1)
        self.fc6 = nn.Linear(hl1, self.img_dim)

    def encode(self, x):
        h1 = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc31(h1), self.fc32(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc5(F.relu(self.fc4(z))))
        return torch.sigmoid(self.fc6(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar