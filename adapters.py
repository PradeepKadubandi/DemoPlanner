import numpy as np

def demopl_v1_data_adapter(data):
    x_dim = 2
    img_res = 32
    img_size = img_res * img_res
    return data[:, 2*x_dim:2*x_dim+img_size] / 255

def demopl_v1_data_to_img(data, batch_size):
    x_dim = 2
    img_res = 32
    return 255 * data.view(batch_size, img_res, img_res)

def demopl_v2_data_adapter(data):
    return demopl_v1_data_adapter(data)

def demopl_v2_data_to_img(data, batch_size):
    return demopl_v1_data_to_img(data, batch_size)