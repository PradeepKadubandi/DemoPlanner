import torch
import torch.nn.functional as F

from adapters import *

class PolicyEvaluator():
    def __init__(self, net, distance_func, label_adapter):
        self.net = net
        self.distance_func = distance_func
        self.label_adapter = label_adapter

    # def rollout(self, x0, yt, k):
    #     xt = x0
    #     xhat_tplus = torch.zeros((k, len(torch.squeeze(x0))))
    #     ut_pred = torch.zeros((k, 2))
    #     dyn_pred = torch.zeros((k, 2))
    #     for i in range(k):
    #         ut = self.policy(torch.cat((xt, yt), dim=1))
    #         if self.discrete_actions:
    #             ux = torch.argmax(ut[:, :3], dim=1, keepdims=True)
    #             uy = torch.argmax(ut[:, 3:6], dim=1, keepdims=True)
    #             ut = torch.cat((ux, uy), dim=1).float()
    #         ut_pred[i] = ut
    #         # dynamics expects ut from [0, 1] but policy argmax results in index [0, 2]
    #         ut = ut / 2
    #         pred = self.dynamics(torch.cat((xt, ut), dim=1))
    #         dyn_pred[i] = pred
    #         if self.using_grad_dynamics:
    #             # first unscale pred from [0, 1] to [-1, 1] and xt from [0, 1] to [0, 32]
    #             pred = (pred * 2.0) - 1.0
    #             xt = xt + (pred / 32)  # This is same as ((xt * 32) + pred) / 32
    #         else:
    #             # The prediction itself is xt_plus and already in the range [0,1], so just need to use that for next step.
    #             xt = pred
    #         xhat_tplus[i] = xt * 32
    #     return xhat_tplus, ut_pred, dyn_pred

    def eval_single_trajectory(self, trajectory):
        with torch.no_grad():
            predictions = self.net.rollout(trajectory)
        ground_truth = self.label_adapter(trajectory)
        step_error = self.distance_func(predictions, ground_truth)
        goal_error = self.distance_func(predictions[-1], ground_truth[-1])
        return goal_error, step_error, ground_truth, predictions

def eval_policy_accuracy(predictions, ground_truth):
    correct_directions = torch.sum(torch.where(ground_truth==predictions, torch.ones_like(ground_truth), torch.zeros_like(ground_truth)), dim=1)
    # print (correct_directions.shape)
    # print (correct_directions[:10])
    correct_labels = torch.where(correct_directions == 2, torch.ones_like(correct_directions), torch.zeros_like(correct_directions))
    # print (correct_labels.shape)
    accuracy = 100 * torch.sum(correct_labels).float() / len(ground_truth)
    return accuracy