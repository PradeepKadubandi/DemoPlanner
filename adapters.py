import numpy as np
import torch
import torch.nn.functional as F
#import visualplanningdimensions
from simplereacherdimensions import *

def demopl_v1_data_to_img(data, batch_size):
    return max_pixel * data.view(batch_size, img_res, img_res)

def identity_adapter(data):
    return data

def It_unscaled_adapter(data):
    return data[:, 2*x_dim:2*x_dim+img_size]

def It_scaled_adapter(data):
    return data[:, 2*x_dim:2*x_dim+img_size] / max_pixel

def It_unscaled_HW_adapter(data):
    return data[:, 2*x_dim:2*x_dim+img_size].view([-1, 1, img_res, img_res])

def It_scaled_HW_adapter(data):
    return data[:, 2*x_dim:2*x_dim+img_size].view([-1, 1, img_res, img_res]) / max_pixel

def Xt_unscaled_adapter(data):
    return data[:, :x_dim]

def Xt_scaled_adapter(data):
    return data[:, :x_dim] / img_res

def Xtplus_unscaled_adapter(data):
    return data[:,  xtplus_begin:xtplus_begin+x_dim]

def Xtplus_scaled_adapter(data):
    return data[:,  xtplus_begin:xtplus_begin+x_dim] / img_res

def Yt_unscaled_adapter(data):
    return data[:, x_dim:x_dim+y_dim]

def Yt_scaled_adapter(data):
    return data[:, x_dim:x_dim+y_dim] / img_res

def Ut_unscaled_adapter(data):
    return data[:, u_begin:u_begin+u_dim]

def Ut_scaled_adapter(data):
    return (data[:, u_begin:u_begin+u_dim] + 1.0) / 2

def XtYt_unscaled_adapter(data):
    return data[:, :x_dim+y_dim]

def XtYt_scaled_adapter(data):
    return data[:, :x_dim+y_dim] / img_res

def dynamics_input_adapter(data):
    scaled_xt = Xt_scaled_adapter(data)
    scaled_ut = Ut_scaled_adapter(data)
    return torch.cat((scaled_xt, scaled_ut), axis=1)

def dynamics_ground_truth_adapter(data):
    return  data[:, xtplus_begin:xtplus_begin+x_dim] / img_res

def dynamics_gradient_ground_truth_adapter(data):
    diff = data[:, xtplus_begin:xtplus_begin+x_dim] - data[:, :x_dim]
    return (diff + 1.0) / 2

def dynamics_grad_gt_X_unscaled(data):
    diff = data[:, xtplus_begin:xtplus_begin+1] - data[:, :1]
    return diff

def dynamics_grad_gt_Y_unscaled(data):
    diff = data[:, xtplus_begin+1:xtplus_begin+2] - data[:, 1:2]
    return diff

def dynamics_grad_gt_X_scaled(data):
    diff = data[:, xtplus_begin:xtplus_begin+1] - data[:, :1]
    return (diff + 1.0) / 2

def dynamics_grad_gt_Y_scaled(data):
    diff = data[:, xtplus_begin+1:xtplus_begin+2] - data[:, 1:2]
    return (diff + 1.0) / 2

def policy_input_adapter(data):
    return data[:, :x_dim+y_dim] / img_res

def Xt_XtYt_scaled_adapter(data):
    '''
    The name is not a typo, the idea behind this is to augment Y with X
    Policy takes as input X and this new Augmented Y
    (This is for training a combination of latent from image and policy together)
    '''
    return torch.cat((data[:, :x_dim], data[:, :x_dim+y_dim]), dim=1) / img_res

def policy_groud_truth_adapter(data):
    '''
    Useful for regression loss
    '''
    return data[:, u_begin:u_begin+u_dim]

def policy_groud_truth_class_adapter(data):
    '''
    Useful for cross entropy loss
    '''
    u = torch.tensor(data[:, u_begin:u_begin+u_dim] + 1.0, dtype=torch.long).detach()
    return u

def policy_groud_truth_one_hot_adapter(data):
    '''
    Useful for binary cross entropy loss
    '''
    u = torch.tensor(data[:, u_begin:u_begin+u_dim] + 1.0, dtype=torch.long).detach()
    oh = F.one_hot(u, num_classes=3)
    return oh.reshape(u.size()[0], -1)

def demopl_v1_filter_and_convert_to_image(data, batch_size):
    return demopl_v1_data_to_img(It_scaled_adapter(data), batch_size)

def multinet_ground_truth_adapter(data):
    '''
    Use this if returning concatenated output from network.
    '''
    It = It_scaled_adapter(data)
    Yt = XtYt_scaled_adapter(data)
    return torch.cat((It, Yt, It, Yt),dim=1)