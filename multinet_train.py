import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from io import StringIO
from datetime import datetime

from utils import writeline
from adapters import *

class SpecialExptRunner(ExptRunnerBase):
    def __init__(self, expt_prefix, net,
                 train_data, test_data, device=None):
        super(SpecialExptRunner, self).__init__(expt_prefix, net, train_data, test_data, device)
        self.combinations = {
            0:['I', 'I'],
            1:['E', 'E'],
            2:['I', 'E'],
            3:['E', 'I']
        }
        self.variations = len(self.combinations)

    def __preTrain(self):
        start = time.time()
        self.net.train()
        self.log_network_details()
        builder = StringIO()
        return start, builder
    
    def __postTrain(self, start, builder):
        writeline(builder, 'Total time taken for training {} sec.'.format(time.time() - start))
        with open(self.log_folder + '/train_log.txt', 'w') as f:
            f.write(builder.getvalue())
        builder.close()

    def __adaptData(self, comb, data):
        ip_adapter = It_scaled_adapter if comb[0] == 'I' else XtYt_scaled_adapter
        gt_adapter = It_scaled_adapter if comb[1] == 'I' else XtYt_scaled_adapter
        ip = ip_adapter(data)
        gt = gt_adapter(data)
        return ip, gt

    def log_train_loss_end_of_epoch(self, eIndex, avg_loss, builder):
        for cI, comb in self.combinations.items():
            writeline(builder, 'Average Training Loss for epoch {}, for combination {}: {}'.format(eIndex, comb, avg_loss[cI]))

    def eval_end_of_epoch(self, eIndex, crit, builder):
        with torch.no_grad():
            for i, comb in self.combinations.items():
                self.net.setNetworksForForward(*comb)
                ip, gt = self.__adaptData(comb, self.test_data)
                op = self.net(ip)
                loss = crit(op, gt)
                writeline(builder, 'Loss on test set after epoch {}, for combination {}: {}'.format(eIndex, comb, loss.item()))

    def eval_end_of_training(self, crit=nn.L1Loss()):
        self.net.eval()
        builder = StringIO()
        with torch.no_grad():
            for i, comb in self.combinations.items():
                self.net.setNetworksForForward(*comb)
                writeline(builder, '-------------------------------------------------------')
                writeline(builder, '                Combination {}'.format(comb))
                writeline(builder, '-------------------------------------------------------')
                ip, gt = self.__adaptData(comb, self.test_data)
                op = self.net(ip)
                loss = crit(op, gt)
                writeline(builder, 'Test Loss For Combination {}: {}'.format(comb, loss.item()))
                for j in range(5):
                    loss = crit(op[j], gt[j])
                    writeline(builder, 'Datapoint {} test error: {}'.format(j, loss.item()))
                    if comb[1] == 'I':
                        batch_size = 1
                        self.save_matplotlib_comparison(batch_size, 
                            demopl_v1_data_to_img(gt[j], batch_size),
                            demopl_v1_data_to_img(op[j], batch_size),
                            filename='final_reconstruction_comb_{}_sample_{}'.format(i, j),
                            printHeader="Final Reconstruction For Test Image {} In Combination {}".format(j, i))

        with open(self.log_folder + '/test_samples.txt', 'w') as f:
            f.write(builder.getvalue())
        builder.close()

    def train2(self, epochs):
        '''
        Train each combinations for epochs one after other.
        Each combination has it's own optimizer.
        '''
        pass

    def train3(self, epochs, crit=None):
        '''
        For each epoch, train all combinations in order.
        Each combination has it's own optimizer.
        '''
        opts = [optim.Adam(self.net.parameters()) for i in range(self.variations)]
        crit = nn.MSELoss().to(self.device) if crit is None else crit
        start,builder = self.__preTrain()

        running_loss = np.zeros((self.variations))
        running_count = np.zeros((self.variations), dtype=np.int32)
        for e in range(epochs):
            running_loss[:] = 0.0
            running_count[:] = 0
            for cIndex, comb in self.combinations.items():
                for i, data in enumerate(self.train_loader, 0):
                    self.net.setNetworksForForward(*comb)
                    ip, gt = self.__adaptData(comb, data)
                    op = self.net(ip)
                    loss = crit(op, gt)
                    opt = opts[cIndex]
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    running_loss[cIndex] += loss.item()
                    running_count[cIndex] += 1
                    index = e * len(self.train_loader) + i
                    self.writer.add_scalar('training_loss_{}'.format(cIndex), loss.item(), index)
            self.log_train_loss_end_of_epoch(e, running_loss / running_count, builder)
            self.eval_end_of_epoch(e, crit, builder)

        self.__postTrain(start, builder)
        torch.save({
            'epoch': epochs,
            'model': self.net,
            'state_dict': self.net.state_dict(),
            'optimizers': opts,
            'criterion': crit
        }, self.checkpoint_file)
        self.eval_end_of_training()

    def train1(self, epochs, opt=None, crit=None):
        '''
        Train all combinations alternating between mini-batches
        Optimizer is common across all combinations.
        '''
        opt = optim.Adam(self.net.parameters()) if opt is None else opt
        crit = nn.MSELoss().to(self.device) if crit is None else crit
        start,builder = self.__preTrain()

        running_loss = np.zeros((self.variations))
        running_count = np.zeros((self.variations), dtype=np.int32)
        for e in range(epochs):
            running_loss[:] = 0.0
            running_count[:] = 0
            for i, data in enumerate(self.train_loader, 0):
                cIndex = i%self.variations
                comb = combinations[cIndex]
                self.net.setNetworksForForward(*comb)
                ip, gt = self.__adaptData(comb, data)
                op = self.net(ip)
                loss = crit(op, gt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running_loss[cIndex] += loss.item()
                running_count[cIndex] += 1
                index = e * len(self.train_loader) + i
                self.writer.add_scalar('training_loss_{}'.format(cIndex), loss.item(), index)
            self.log_train_loss_end_of_epoch(e, running_loss / running_count, builder)
            self.eval_end_of_epoch(e, crit, builder)

        self.__postTrain(start, builder)
        torch.save({
            'epoch': epochs,
            'model': self.net,
            'state_dict': self.net.state_dict(),
            'optim': opt,
            'optim_state_dict': opt.state_dict(),
            'criterion': crit
        }, self.checkpoint_file)
        self.eval_end_of_training()