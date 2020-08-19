import numpy as np
import torch
import torch.nn.functional as F
from simplereacherdimensions import *

def It_unscaled_adapter(data):
    return data[images_key].float()

def It_scaled_adapter(data):
    return data[images_key].float() / max_pixel

def It_scaled_adapter_eval(data):
    return data.float() / max_pixel

def It_unscaled_HW_adapter(data):
    return data[images_key].float().view([-1, 1, img_res, img_res])

def It_scaled_HW_adapter(data):
    return data[images_key].float().view([-1, 1, img_res, img_res]) / max_pixel

def Xt_unscaled_adapter(data):
    return data[states_key][:, :x_dim].float()

def Xt_scaled_adapter(data):
    return (data[states_key][:, :x_dim] + joint_max).float() / (2 * joint_max)

def Yt_unscaled_adapter(data):
    return data[states_key][:, x_dim:x_dim+y_dim].float()

def Yt_scaled_adapter(data):
    return (data[states_key][:, x_dim:x_dim+y_dim] + joint_max).float() / (2 * joint_max)

def Ut_unscaled_adapter(data):
    return data[states_key][:, u_begin:u_begin+u_dim].float()

def Ut_scaled_adapter(data):
    return (data[states_key][:, u_begin:u_begin+u_dim] + action_max).float() / (2 * action_max)

def Ut_scaled_adapter_reverse(net_output):
    return (net_output * (2 * action_max)) - action_max

def Ut_scaled_adapter_zero_center(data):
    return (data[states_key][:, u_begin:u_begin+u_dim]).float() / (action_max)

def Ut_scaled_adapter_zero_center_reverse(net_output):
    return net_output * action_max

def XtYt_unscaled_adapter(data):
    return data[states_key][:, :x_dim+y_dim].float()

def XtYt_scaled_adapter(data):
    return (data[states_key][:, :x_dim+y_dim] + joint_max).float() / (2 * joint_max)

def XtYt_scaled_adapter_eval(sim_state):
    return (sim_state[:, :x_dim+y_dim] + joint_max).float() / (2 * joint_max)

def identity_adapter(data):
    return data
