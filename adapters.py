import numpy as np
import torch

def demopl_v1_data_adapter(data):
    x_dim = 2
    img_res = 32
    img_size = img_res * img_res
    return data[:, 2*x_dim:2*x_dim+img_size] / 255

def demopl_v1_data_to_img(data, batch_size):
    x_dim = 2
    img_res = 32
    return 255 * data.view(batch_size, img_res, img_res)

def dynamics_input_adapter(data):
    x_dim = 2
    u_dim = 2
    img_res = 32
    img_size = img_res * img_res
    u_begin = 2*x_dim + img_size
    normalized_xt = data[:, :2] / 32
    normalized_ut = (data[:, u_begin:u_begin+u_dim] + 1.0) / 2
    return torch.cat((normalized_xt, normalized_ut), axis=1)

def dynamics_ground_truth_adapter(data):
    x_dim = 2
    u_dim = 2
    img_res = 32
    img_size = img_res * img_res
    xtplus_begin = 2*x_dim + img_size + u_dim
    return  data[:, xtplus_begin:xtplus_begin+x_dim] / 32

def dynamics_gradient_ground_truth_adapter(data):
    x_dim = 2
    u_dim = 2
    img_res = 32
    img_size = img_res * img_res
    xtplus_begin = 2*x_dim + img_size + u_dim
    diff = data[:, xtplus_begin:xtplus_begin+x_dim] - data[:, :x_dim]
    return (diff + 1.0) / 2