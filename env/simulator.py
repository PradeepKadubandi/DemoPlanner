import env
from config import reacher
from config import argparser
from simplereacherdimensions import *
import gym
from env.reacher.simple_reacher import SimpleReacherEnv
from matplotlib.backends.backend_pdf import PdfPages
import torch
import os
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt

class simulator:
    def __init__(self):
        parser = argparser()
        args, unparsed = parser.parse_known_args()
        reacher.add_arguments(parser)
        args, unparsed = parser.parse_known_args()
        self.env = gym.make(args.env, **args.__dict__)

    def reset(self, initialState):
        self.env.reset()
        initial_pos = self.env.sim.data.qpos
        initial_pos[:x_dim+y_dim] = initialState[:x_dim+y_dim].clone()
        self.env.set_state(initial_pos, self.env.sim.data.qvel)
        self.env.visualize_goal_indicator(initialState[x_dim:x_dim+y_dim].clone())
        self.env.visualize_dummy_indicator(initialState[x_dim:x_dim+y_dim].clone())

    def step(self, action):
        self.env.step(action, is_planner=False)
        return self.getState()

    def getState(self):
        return torch.as_tensor(self.env.sim.data.qpos[:x_dim+y_dim].copy()).float(), self.render_image()
        
    def render_image(self):
        img = self.env.render('rgb_array') * 255.0
        low = (500 - img_res) // 2
        high = low + img_res
        cropped_rgb = img[low:high, low:high, :].astype(np.uint8)
        return torch.as_tensor(cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)).float()

class evaluator:
    def __init__(self, chkpt_file, data, device, policy_takes_image, persist_to_disk):
        self.log_folder = os.path.dirname(chkpt_file)
        self.chkpt = torch.load(self.log_folder + '/train_checkpoint.tar', map_location=device)
        self.net = self.chkpt['model']
        self.data = data
        self.policy_takes_image = policy_takes_image
        self.persist_to_disk = persist_to_disk
        self.sim = simulator()

    def rollout(self, gt_trajectory):
        gt_states, gt_images = gt_trajectory[states_key], gt_trajectory[images_key]
        self.sim.reset(gt_states[0])
        curr_state, curr_image = self.sim.getState()
        states = torch.as_tensor(curr_state.reshape(1, -1))
        images = torch.as_tensor(curr_image.reshape(1, -1))
        actions = None
        for i in range(len(gt_states)-1):
            with torch.no_grad():
                action = self.net(curr_image if self.policy_takes_image else curr_state)
                if actions is None:
                    actions = torch.as_tensor(action.reshape(1, -1))
                else:
                    actions = torch.cat((actions, action.reshape(1, -1)), dim=0)
            curr_state, curr_image = self.sim.step(action.numpy())
            states = torch.cat((states, curr_state.reshape(1, -1)), dim=0)
            images = torch.cat((images, curr_image.reshape(1, -1)), dim=0)
        actions = torch.cat((actions, torch.zeros_like(actions[0]).reshape(1, -1)), dim=0)
        states = torch.cat((states, actions), dim=1)
        return {states_key: states, images_key: images}

    def evaluate(self):
        N = len(self.data)
        errors = torch.zeros((N, 6))
        allLabels = []
        allPredictions = []
        for index in range(N):
            labels = self.data[index]
            predictions = self.rollout(labels)
            allLabels.append(labels)
            allPredictions.append(predictions)
            label_states, label_images = labels[states_key], labels[images_key]
            pred_states, pred_images = predictions[states_key], predictions[images_key]
            errors[index][0] = index
            errors[index][1] = len(label_states)
            errors[index][2] = F.l1_loss(pred_states[:, u_begin:], label_states[:, u_begin:])
            errors[index][3] = F.l1_loss(pred_states[:, :x_dim], label_states[:, :x_dim])
            errors[index][4] = F.l1_loss(pred_states[-1, :x_dim+y_dim], label_states[-1, :x_dim+y_dim])
            errors[index][5] = F.l1_loss(pred_states[:, :x_dim+y_dim], label_states[:, :x_dim+y_dim])
        
        aggregate_row_headers = ['Sum', 'Average', 'Min Index', 'Min Value', 'Max Index', 'Max Value', 'Counts']
        aggregates = torch.zeros((7, 6))
        aggregates[0] = torch.sum(errors, dim=0)
        aggregates[1] = aggregates[0] / N
        aggregates[3], aggregates[2] = torch.min(errors, dim=0)
        aggregates[5], aggregates[4] = torch.max(errors, dim=0)
        lower_threshholds = torch.Tensor([-1, 10]) # Total trajectories, trajectories of len > 10
        upper_threshholds = torch.Tensor([0.01, 0.01, 0.01, 0.01]) # trajectories of policy loss, state loss, dynamics step loss, goal loss < 0.01
        aggregates[6] = torch.sum(torch.cat((
            torch.gt(errors[:, :2], lower_threshholds),
            torch.lt(errors[:, 2:], upper_threshholds)), dim=1), dim=0)
        agg = aggregates.numpy()
        
        result_folder = self.log_folder + '/Evaluations'
        if self.persist_to_disk:
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)

            with open(result_folder + '/trajectory_eval_metrics.csv', 'a') as f:
                f.write('Trajectory Index,Trajectory Length,Policy Accuracy,Policy L1 Loss,Dynamics Loss (step),Goal Deviation\n')
                f.write('Aggregates:\n')
                for i in range(len(aggregate_row_headers)):
                    f.write(aggregate_row_headers[i])
                    f.write(',')
                    f.write(str.join(',', [str(val) for val in agg[i, 1:]]))
                    f.write('\n')
                f.write('\nInidividual:\n')
                np.savetxt(f, errors.numpy(), fmt='%5.2f', delimiter=',')
        
        sample_descriptions = ['Least Policy L1 Error',
                            'Least Trajectory Error','Least Goal Error',
                            'Highest Policy L1 Error',
                            'Highest Trajectory Error','Highest Goal Error']
        # For policy, min means worst, for other errors, min means best
        indices = torch.cat((aggregates[2, 2:5], aggregates[4, 2:5])).long()
        values = torch.cat((aggregates[3, 2:5], aggregates[5, 2:5]))
        print ('-------------------------------------------------------------')
        print (self.log_folder)
        print ('-------------------------------------------------------------')
        print ('Number of Trajectories Ended up Within Goal Region: {}'.format(aggregates[6, 5].item()))
        print ('Average Policy L1 Loss: {}'.format(aggregates[1, 2].item()))
        print ('Average State Loss: {}'.format(aggregates[1, 3].item()))
        print ('Average Trajectory Loss: {}'.format(aggregates[1, 4].item()))
        print ('Average Goal Loss: {}'.format(aggregates[1, 5].item()))
        print ('')
        for i in range(len(sample_descriptions)):
            print ('{}: Index = {}, Value = {}'.format(sample_descriptions[i], indices[i], values[i]))
            if i == 2:
                print ('')

        if self.persist_to_disk:
            for i in range(len(sample_descriptions)):
                index = indices[i]
                labels = allLabels[index][images_key]
                predictions = allPredictions[index][images_key]
                filename = '/traj_' + str(index.item()) + '_' + str.join('_', str.split(sample_descriptions[i])) + '.pdf'
                with PdfPages(result_folder + filename) as pdf:
                    for i in range(len(labels)):
                        fig = plt.figure()
                        plt.subplot(1,2,1)
                        plt.imshow(labels[i, :].reshape(img_res,img_res), cmap=plt.get_cmap("gray"))
                        plt.subplot(1,2,2)
                        plt.imshow(predictions[i, :].reshape(img_res,img_res), cmap=plt.get_cmap("gray"))
                        pdf.savefig(fig)
                        plt.close(fig)
