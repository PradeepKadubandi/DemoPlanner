import torch
import torch.nn.functional as F

from adapters import *

class PolicyEvaluator():
    def __init__(self, distance_func, discrete_actions=True,
                 policy_checkpoint=None, dynamics_checkpoint=None,
                 policy=None, dynamics=None,
                #  pol_ip_adapter=None,
                #  pol_lb_adapter=None,
                #  dyn_ip_adapter=None,
                #  dyn_lb_adapter=None,
                 using_grad_dynamics=True):
        self.distance_func = distance_func
        self.discrete_actions = discrete_actions
        self.policy_checkpoint = policy_checkpoint
        self.dynamics_checkpoint = dynamics_checkpoint
        self.policy = policy if policy else policy_checkpoint['model']
        self.dynamics = dynamics if dynamics else dynamics_checkpoint['model']
        # self.pol_ip_adapter = pol_ip_adapter if pol_ip_adapter else policy_checkpoint['input_adapter']
        # self.pol_lb_adapter = pol_lb_adapter if pol_lb_adapter else policy_checkpoint['label_adapter']
        # self.dyn_ip_adapter = dyn_ip_adapter if dyn_ip_adapter else dynamics_checkpoint['input_adapter']
        # self.dyn_lb_adapter = dyn_lb_adapter if dyn_lb_adapter else dynamics_checkpoint['label_adapter']
        self.using_grad_dynamics = using_grad_dynamics

    def rollout(self, x0, yt, k):
        xt = x0
        xhat_tplus = torch.zeros((k, len(torch.squeeze(x0))))
        ut_pred = torch.zeros((k, 2))
        dyn_pred = torch.zeros((k, 2))
        for i in range(k):
            ut = self.policy(torch.cat((xt, yt), dim=1))
            if self.discrete_actions:
                ux = torch.argmax(ut[:, :3], dim=1, keepdims=True)
                uy = torch.argmax(ut[:, 3:6], dim=1, keepdims=True)
                ut = torch.cat((ux, uy), dim=1).float()
            ut_pred[i] = ut
            # dynamics expects ut from [0, 1] but policy argmax results in index [0, 2]
            ut = ut / 2
            pred = self.dynamics(torch.cat((xt, ut), dim=1))
            dyn_pred[i] = pred
            if self.using_grad_dynamics:
                # first unscale pred from [0, 1] to [-1, 1] and xt from [0, 1] to [0, 32]
                pred = (pred * 2.0) - 1.0
                xt = xt + (pred / 32)  # This is same as ((xt * 32) + pred) / 32
            else:
                # The prediction itself is xt_plus and already in the range [0,1], so just need to use that for next step.
                xt = pred
            xhat_tplus[i] = xt * 32
        return xhat_tplus, ut_pred, dyn_pred

    def eval_single_trajectory(self, trajectory):
        with torch.no_grad():
            xtplus = Xtplus_unscaled_adapter(trajectory)
            xhat_tplus = torch.zeros_like(xtplus)
            x0 = Xt_scaled_adapter(trajectory[:1])
            yt = Yt_scaled_adapter(trajectory[:1])
            xhat_tplus, ut_pred, dyn_pred = self.rollout(x0, yt, len(trajectory))
        step_error = self.distance_func(xhat_tplus, xtplus)
        goal_error = self.distance_func(xhat_tplus[-1], xtplus[-1])
        return goal_error, step_error, xhat_tplus, xtplus, ut_pred, dyn_pred