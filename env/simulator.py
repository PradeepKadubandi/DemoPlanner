import env
from config import reacher
from config import argparser
from simplereacherdimensions import *
import gym
from env.reacher.simple_reacher import SimpleReacherEnv
import torch

class simulator:
    def __init__(self, initialState, is_state_image = False):
        self.is_state_image = is_state_image
        parser = argparser()
        args, unparsed = parser.parse_known_args()
        reacher.add_arguments(parser)
        args, unparsed = parser.parse_known_args()
        self.env = gym.make(args.env, **args.__dict__)
        self.env.reset()
        self.env.set_state(initialState.clone(), self.env.sim.data.qvel)
        self.env.visualize_goal_indicator(initialState[x_dim:x_dim+y_dim].clone())
        self.env.visualize_dummy_indicator(initialState[x_dim:x_dim+y_dim].clone())

    def step(self, action):
        self.env.step(action, is_planner=True)
        return self.getState()

    def getState(self):
        if self.is_state_image:
            return self.render_image()
        return torch.as_tensor(self.env.sim.data.qpos[:x_dim+y_dim].copy())
        
    def render_image(self):
        img = self.env.render('rgb_array') * 255.0
        low = (500 - img_res) // 2
        high = low + img_res
        cropped_rgb = img[low:high, low:high, :].astype(np.uint8)
        return torch.as_tensor(cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY))

class evaluator:
    def __init__(self, net, is_state_image=False):
        self.net = net
        self.is_state_image = is_state_image

    def rollout(self, gt_trajectory):
        if isinstance(gt_trajectory, dict):
            gt_trajectory = gt_trajectory[states_key]
        sim = simulator(gt_trajectory[0], self.is_state_image)
        curr_state = sim.getState()
        result = torch.as_tensor(curr_state.reshape(1, -1))
        for i in range(len(gt_trajectory)-1):
            with torch.no_grad():
                action = self.net(curr_state)
            curr_state = sim.step(action.numpy())
            result = torch.cat((result, curr_state.reshape(1, -1)), dim=0)
        return result