import torch
import torch.nn as nn
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# TODO: Not sure if it's a better idea to move these functions into network implementation itself
# If we ever use one loss function with one network, then above is better
# Instead, if we ever combine different loss functions with different network structures, current model is better.
def vae_loss_adapter(net, ip_batch):
    recon_batch, mu, logvar = net(ip_batch)
    return recon_batch, vae_loss(recon_batch, ip_batch, mu, logvar)

def mse_loss_adapter(net, ip_batch):
    op_batch = net(ip_batch)
    return op_batch, F.mse_loss(ip_batch, op_batch)

def smooth_l1_loss_adapter(net, ip_batch):
    op_batch = net(ip_batch)
    return op_batch, F.smooth_l1_loss(ip_batch, op_batch)