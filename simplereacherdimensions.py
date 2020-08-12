import numpy as np

x_dim = 3
y_dim = 3
u_dim = 3
img_res = 256
img_size = img_res * img_res
max_pixel = 255
joint_max = np.pi
action_max = 0.2

u_begin = x_dim + y_dim
images_key = 'images'
states_key = 'states'
