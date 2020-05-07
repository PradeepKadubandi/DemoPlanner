import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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

def mse_loss_adapter(net, ip_batch, labels=None):
    labels = ip_batch if labels is None else labels
    op_batch = net(ip_batch)
    return op_batch, F.mse_loss(labels, op_batch)

def smooth_l1_loss_adapter(net, ip_batch, labels=None):
    labels = ip_batch if labels is None else labels
    op_batch = net(ip_batch)
    return op_batch, F.smooth_l1_loss(labels, op_batch)

def l1_loss_adapter(net, ip_batch, labels=None):
    labels = ip_batch if labels is None else labels
    op_batch = net(ip_batch)
    return op_batch, F.l1_loss(labels, op_batch)

def control_ce_loss_adapter(net, ip_batch, labels=None):
    '''
    dim(labels) = dim(op_batch) = (batch_size, 6)
    loss is calculated separately for first 3 and second half
    '''
    labels = ip_batch if labels is None else labels
    op_batch = net(ip_batch)
    loss = F.cross_entropy(op_batch[:, :3], labels[:, 0])
    loss += F.cross_entropy(op_batch[:, 3:], labels[:, 1])
    return op_batch, loss

def bce_loss_adapter(net, ip_batch, labels=None):
    labels = ip_batch if labels is None else labels
    op_batch = net(ip_batch)
    return op_batch, F.binary_cross_entropy(op_batch, labels)

def vae_smooth_l1_loss_adapter(net, ip_batch):
    op_batch, _, _ = net(ip_batch)
    return op_batch, F.smooth_l1_loss(ip_batch, op_batch)

def ImageEnvEncoder_mse_loss_adapter(net, ip_batch, labels=None):
    labels = ip_batch if labels is None else labels
    I = It_scaled_adapter(labels)
    y = XtYt_scaled_adapter(labels)
    z_img, z_env, I_recon, y_recon = net(ip_batch)
    loss = F.mse_loss(z_env, z_img) + F.mse_loss(I_recon, I) + F.mse_loss(y_recon, y)
    return I_recon, loss

def ImageEnvEncoder_l1_loss_adapter(net, ip_batch, labels=None, reduction='mean'):
    labels = ip_batch if labels is None else labels
    I = It_scaled_adapter(labels)
    y = XtYt_scaled_adapter(labels)
    z_img, z_env, I_recon, y_recon = net(ip_batch)
    if reduction == 'none':
        loss = torch.cat((
            F.l1_loss(z_env, z_img, reduction=reduction),
            F.l1_loss(I_recon, I, reduction=reduction),
            F.l1_loss(y_recon, y, reduction=reduction),
        ), dim=1)
    else:
        loss = F.l1_loss(z_env, z_img) + F.l1_loss(I_recon, I) + F.l1_loss(y_recon, y)
    return I_recon, loss

def MultiEncoderNet_l1_loss_adapter(net, data, labels=None, reduction='mean'):
    labels = data if labels is None else labels
    I = It_scaled_adapter(labels)
    Y = XtYt_scaled_adapter(labels)
    I_to_I, I_to_Y, Y_to_I, Y_to_Y = net(data)
    if reduction == 'none':
        loss = torch.cat((
            F.l1_loss(I_to_I, I, reduction=reduction),
            F.l1_loss(I_to_Y, Y, reduction=reduction),
            F.l1_loss(Y_to_I, I, reduction=reduction),
            F.l1_loss(Y_to_Y, Y, reduction=reduction)
        ), dim=1)
    else:
        loss = F.l1_loss(I_to_I, I) + F.l1_loss(I_to_Y, Y) + F.l1_loss(Y_to_I, I) + F.l1_loss(Y_to_Y, Y)
    return I_to_I, loss

def MultiEncoderNet_mse_loss_adapter(net, data, labels=None):
    labels = data if labels is None else labels
    I = It_scaled_adapter(labels)
    Y = XtYt_scaled_adapter(labels)
    I_to_I, I_to_Y, Y_to_I, Y_to_Y = net(data)
    loss = F.mse_loss(I_to_I, I) + F.mse_loss(I_to_Y, Y) + F.mse_loss(Y_to_I, I) + F.mse_loss(Y_to_Y, Y)
    return I_to_I, loss

def EndToEndEnv_EndToEnd_mse_loss_adapter(net, ip_batch, labels=None):
    labels = ip_batch if labels is None else labels
    # y_t = XtYt_scaled_adapter(labels)[:, 2:]
    # u_t = Ut_scaled_adapter(labels)
    diff_t = dynamics_gradient_ground_truth_adapter(labels)
    yhat_t, policy_output, uhat_t, dynamics_out = net(ip_batch)
    # loss = F.mse_loss(yhat_t, y_t) + F.mse_loss(uhat_t, u_t) + F.mse_loss(dynamics_out, diff_t)
    loss = F.mse_loss(dynamics_out, diff_t)
    return dynamics_out, loss

def EndToEndEnv_EndToEnd_l1_loss_adapter(net, ip_batch, labels=None, reduction='mean'):
    labels = ip_batch if labels is None else labels
    # y_t = XtYt_scaled_adapter(labels)[:, 2:]
    # u_t = Ut_scaled_adapter(labels)
    diff_t = dynamics_gradient_ground_truth_adapter(labels)
    yhat_t, policy_output, uhat_t, dynamics_out = net(ip_batch)
    if reduction == 'none':
        loss = F.l1_loss(dynamics_out, diff_t, reduction=reduction)
    else:
        loss = F.l1_loss(dynamics_out, diff_t)
    return dynamics_out, loss

def EndToEndEnv_Policy_ce_loss_adapter(net, ip_batch, labels=None):
    labels = ip_batch if labels is None else labels
    # y_t = XtYt_scaled_adapter(labels)[:, 2:]
    u_t = policy_groud_truth_class_adapter(labels).to(device)
    diff_t = dynamics_gradient_ground_truth_adapter(labels).to(device)
    yhat_t, policy_output, uhat_t, dynamics_out = net(ip_batch)
    # loss = F.mse_loss(yhat_t, y_t) + F.mse_loss(uhat_t, u_t) + F.mse_loss(dynamics_out, diff_t)
    loss = F.cross_entropy(policy_output[:, :3], u_t[:, 0])
    loss += F.cross_entropy(policy_output[:, 3:], u_t[:, 1])
    return policy_output, loss