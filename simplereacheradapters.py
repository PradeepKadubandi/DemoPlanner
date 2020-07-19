import numpy as np
import torch
import torch.nn.functional as F
from simplereacherdimensions import *

def It_unscaled_adapter(data):
    return data[images_key].float()

def It_scaled_adapter(data):
    return data[images_key].float() / max_pixel

def It_unscaled_HW_adapter(data):
    return data[images_key].float().view([-1, 1, img_res, img_res])

def It_scaled_HW_adapter(data):
    return data[images_key].float().view([-1, 1, img_res, img_res]) / max_pixel

def Xt_unscaled_adapter(data):
    return data[states_key][:, :x_dim]

def Xt_scaled_adapter(data):
    return (data[states_key][:, :x_dim] + joint_max) / (2 * joint_max)