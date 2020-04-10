import numpy as np
import matplotlib.pyplot as plt
import time
import os
import glob

import torch
import torch.nn as nn

from adapters import *
from networks.lossfunctions import *

class ReportResults:
    def __init__(self, map_netname_net, testData, trainData, device, data_adapter_func, data_to_label_adapter, loss_adapter_func=None):
        self.map_netname_net = map_netname_net
        self.testData = testData
        self.trainData = trainData
        self.device = device
        self.data_adapter_func = data_adapter_func
        self.data_to_label_adapter = data_to_label_adapter
        self.loss_adapter_func = l1_loss_adapter if loss_adapter_func is None else loss_adapter_func

    def run_mini_batch(self, net, miniBatch):
        ip_batch = self.data_adapter_func(miniBatch)
        ground_truth = self.data_to_label_adapter(miniBatch)
        ground_truth = ground_truth.to(self.device)
        op_batch, loss = self.loss_adapter_func(net, ip_batch, ground_truth)
        return ip_batch, op_batch, loss

    def enumerate_files(self, rootdir='runs', extension='tar'):
        for filename in sorted(glob.iglob(rootdir + '/**/*.' + extension, recursive=True)):
            yield filename

    def load_net(self, filename):
        checkpoint = torch.load(filename)
        net = checkpoint['model'].to(self.device)
        net.eval()
        return net

    def build_net(self, filename):
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

    def eval_test_samples(self, net):
        with torch.no_grad():
            rows = min(5, len(self.testData))
            for i in range(rows):
                ip_batch, op_batch, loss = self.run_mini_batch(net,  self.testData[i:i+1]) #Used range expression to force mini-batch dimension
                print('datapoint', str(i), 'test error', loss.item())
                if ip_batch[0].size()[0] == 1024:
                    ip_batch = ip_batch.cpu()
                    op_batch = op_batch.cpu()
                    plt.imshow(ip_batch[0].reshape(32,32), cmap=plt.get_cmap("gray"), vmin=0, vmax=1)
                    plt.show()
                    plt.imshow(op_batch[0].reshape(32,32), cmap=plt.get_cmap("gray"), vmin=0, vmax=1)
                    plt.show()

    def eval_performance_metrics(self, checkpoint_file, net):
        folder = os.path.basename(os.path.dirname(checkpoint_file))
        parts = folder.split('-')
        netname = parts[7]
        _, _, test_loss = self.run_mini_batch(net, self.testData)
        _, _, train_loss = self.run_mini_batch(net, self.trainData)
        
        time = str.join(':', parts[:5])
        learning = parts[5]
        dataset = parts[6]
        losstype = parts[8]
        print('{},{},{},{},{},{},{}'.format(time, learning, dataset, netname, losstype, test_loss.item(), train_loss.item()))

    def generate_csv_comparison_report_old(self, rootdir='runs', extension='pth'):
        print('Time,Goal,DataSet,Network,Loss(Training),TestLoss,TrainLoss')
        for checkpoint_file in self.enumerate_files(rootdir, extension):
            net = self.build_net(checkpoint_file)
            self.eval_performance_metrics(checkpoint_file, net)

    def generate_csv_comparison_report(self, rootdir='runs', extension='tar'):
        print('Time,Goal,DataSet,Network,Loss(Training),TestLoss,TrainLoss')
        for checkpoint_file in self.enumerate_files(rootdir, extension):
            net = self.load_net(checkpoint_file)
            self.eval_performance_metrics(checkpoint_file, net)