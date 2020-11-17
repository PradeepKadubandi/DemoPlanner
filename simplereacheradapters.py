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

def It_scaled_HW_adapter_eval(data):
    return data.float().view([-1, 1, img_res, img_res]) / max_pixel

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

def Xt_XtYt_scaled_adapter(data):
    '''
    The name is not a typo, the idea behind this is to augment Y with X
    Policy takes as input X and this new Augmented Y
    (This is for training a combination of latent from image and policy together)
    '''
    st = data[states_key]
    xt_xtyt = torch.cat((st[:, :x_dim], st[:, :x_dim+y_dim]), dim=1)
    return (xt_xtyt + joint_max).float() / (2 * joint_max)

def Xt_XtYt_scaled_adapter_eval(sim_state):
    xt_xtyt = torch.cat((sim_state[:, :x_dim], sim_state[:, :x_dim+y_dim]), dim=1)
    return (xt_xtyt + joint_max).float() / (2 * joint_max)

def XtYt_trig_adapter(data):
    xtyt = data[states_key][:, :x_dim+y_dim].float()
    trig = torch.cat((torch.cos(xtyt), torch.sin(xtyt)), dim=1)
    return trig

def XtYt_trig_adapter_eval(sim_state):
    xtyt = sim_state[:, :x_dim+y_dim].float()
    trig = torch.cat((torch.cos(xtyt), torch.sin(xtyt)), dim=1)
    return trig

def XtIt_scaled_adapter(data):
    return torch.cat(((data[states_key][:, :x_dim] + joint_max).float() / (2 * joint_max), data[images_key].float() / max_pixel), dim=1)

def XtIt_scaled_adapter_eval(data):
    return torch.cat(((data[:, :x_dim] + joint_max).float() / (2 * joint_max), data[:, x_dim:].float() / max_pixel), dim=1)

def Xt_from_XtIt_adapter(data):
    return data[:, :x_dim]

def It_from_XtIt_adapter(data):
    return data[:, x_dim:]

def identity_adapter(data):
    return data
