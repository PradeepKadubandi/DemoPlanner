import numpy as np
import matplotlib.pyplot as plt
import time
import os

import torch
import torch.nn as nn

from adapters import *
from networks.lossfunctions import *

def load_mapped_state_dict(target, source, tPrefix, sPrefix):
    new_dict = target.state_dict()
    for p,v in source.state_dict().items():
        targetKey = p.replace(sPrefix, tPrefix)
        if targetKey in target.state_dict().keys():
            new_dict[targetKey] = v
    target.load_state_dict(new_dict)

class ReportResults:
    def __init__(self, testData, trainData, device, checkpointFile, useL1Loss=False):
        self.checkpoint_file = checkpointFile
        self.checkpoint = torch.load(checkpointFile, map_location=device)
        self.map_netname_net = None
        self.testData = testData
        self.trainData = trainData
        self.device = device
        self.data_adapter_func = self.checkpoint['input_adapter']
        self.data_to_label_adapter = self.checkpoint['label_adapter']
        self.loss_adapter_func = l1_loss_adapter if useL1Loss else self.checkpoint['loss_adapter']
        self.net = self.__load_net()

    def run_mini_batch(self, net, miniBatch):
        ip_batch = self.data_adapter_func(miniBatch)
        ground_truth = self.data_to_label_adapter(miniBatch) if self.data_to_label_adapter else ip_batch
        ground_truth = ground_truth.to(self.device)
        op_batch, loss = self.loss_adapter_func(net, ip_batch, ground_truth)
        return ip_batch, op_batch, loss

    def __load_net(self):
        net = self.checkpoint['model'].to(self.device)
        net.eval()
        return net

    def build_net(self, filename):
        '''
        Deprecated.
        '''
        if self.map_netname_net is None:
            return None
        folder = os.path.basename(os.path.dirname(filename))
        parts = folder.split('-')
        netname = parts[7]
        if folder in self.map_netname_net.keys():
            net = self.map_netname_net[folder]()
        elif netname in self.map_netname_net.keys():
            net = self.map_netname_net[netname]()
        else:
            return None
        net = net.to(self.device)
        net.load_state_dict(torch.load(filename, map_location=self.device))
        return net

    def eval_test_samples(self):
        with torch.no_grad():
            rows = min(5, len(self.testData))
            for i in range(rows):
                ip_batch, op_batch, loss = self.run_mini_batch(self.net,  self.testData[i:i+1]) #Used range expression to force mini-batch dimension
                print('datapoint', str(i), 'test error', loss.item())
                if op_batch[0].size()[0] == 1024:
                    ip_batch = ip_batch.cpu()
                    op_batch = op_batch.cpu()
                    plt.imshow(ip_batch[0].reshape(32,32), cmap=plt.get_cmap("gray"), vmin=0, vmax=1)
                    plt.show()
                    plt.imshow(op_batch[0].reshape(32,32), cmap=plt.get_cmap("gray"), vmin=0, vmax=1)
                    plt.show()

    def eval_performance_metrics(self):
        folder = os.path.basename(os.path.dirname(self.checkpoint_file))
        parts = folder.split('-')
        netname = parts[7]
        _, _, test_loss = self.run_mini_batch(self.net, self.testData)
        _, _, train_loss = self.run_mini_batch(self.net, self.trainData)
        
        time = str.join(':', parts[:5])
        learning = parts[5]
        dataset = parts[6]
        losstype = parts[8]
        print('{},{},{},{},{},{},{}'.format(time, learning, dataset, netname, losstype, test_loss.item(), train_loss.item()))