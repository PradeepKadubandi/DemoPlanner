import numpy as np
import torch
import torch.nn.functional as F

x_dim = 2
y_dim = 2
u_dim = 2
img_res = 32
img_size = img_res * img_res

u_begin = 2*x_dim + img_size
xtplus_begin = 2*x_dim + img_size + u_dim

def demopl_v1_data_adapter(data):
    return data[:, 2*x_dim:2*x_dim+img_size] / 255

def demopl_v1_data_to_img(data, batch_size):
    return 255 * data.view(batch_size, img_res, img_res)

def dynamics_input_adapter(data):
    normalized_xt = data[:, :2] / 32
    normalized_ut = (data[:, u_begin:u_begin+u_dim] + 1.0) / 2
    return torch.cat((normalized_xt, normalized_ut), axis=1)

def dynamics_ground_truth_adapter(data):
    return  data[:, xtplus_begin:xtplus_begin+x_dim] / 32

def dynamics_gradient_ground_truth_adapter(data):
    diff = data[:, xtplus_begin:xtplus_begin+x_dim] - data[:, :x_dim]
    return (diff + 1.0) / 2

def policy_input_adapter(data):
    return data[:, :x_dim+y_dim] / 32

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