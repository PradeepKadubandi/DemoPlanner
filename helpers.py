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
    def __init__(self, map_netname_net, device, data_adapter_func, data_to_label_adapter, loss_adapter_func=None):
        self.map_netname_net = map_netname_net
        self.device = device
        self.data_adapter_func = data_adapter_func
        self.data_to_label_adapter = data_to_label_adapter
        self.loss_adapter_func = l1_loss_adapter if loss_adapter_func is None else loss_adapter_func

    def run_mini_batch(self, net, miniBatch):
        ip_batch = self.data_adapter_func(miniBatch)
        ground_truth = self.data_to_label_adapter(miniBatch)
        ground_truth = ground_truth.to(self.device)
        op_batch, loss = self.loss_adapter_func(net, ip_batch, ground_truth)
        return op_batch, loss

    def build_net(self, rootdir='runs'):
        for filename in sorted(glob.iglob(rootdir + '/**/*.pth', recursive=True)):
            folder = os.path.basename(os.path.dirname(filename))
            parts = folder.split('-')
            netname = parts[7]
            if folder in self.map_netname_net.keys():
                net = self.map_netname_net[folder]()
            elif netname in self.map_netname_net.keys():
                net = self.map_netname_net[netname]()
            else:
                continue
            net = net.to(self.device)
            net.load_state_dict(torch.load(filename, map_location=self.device))
            yield folder, net

    def show_test_samples(self, data, rootdir='runs'):
        for folder, net in self.build_net(rootdir):
            parts = folder.split('-')
            netname = parts[7]
            # loss_adapter = vae_smooth_l1_loss_adapter if netname.lower().find('vae') >= 0 else smooth_l1_loss_adapter
            with torch.no_grad():
                criterion = nn.L1Loss()
                rows = min(5, len(data))
                op_batch, _ = self.run_mini_batch(net, data[:rows])
                ip_batch = ip_batch.cpu()
                op_batch = op_batch.cpu()
                for i in range(rows):
                    print('datapoint', str(i), 'test error', criterion(input=ip_batch[i], target=op_batch[i]))
                    plt.imshow(ip_batch[i].reshape(32,32), cmap=plt.get_cmap("gray"), vmin=0, vmax=1)
                    plt.show()
                    plt.imshow(op_batch[i].reshape(32,32), cmap=plt.get_cmap("gray"), vmin=0, vmax=1)
                    plt.show()

    def generate_csv_comparison_report(self, testData, trainData, rootdir='runs'):
        print('Time,Goal,DataSet,Network,Loss(Training),TestLoss,TrainLoss')
        for folder, net in self.build_net(rootdir):
            parts = folder.split('-')
            netname = parts[7]
            # loss_adapter = vae_smooth_l1_loss_adapter if netname.lower().find('vae') >= 0 else smooth_l1_loss_adapter
            _, test_loss = self.run_mini_batch(net, testData)
            _, train_loss = self.run_mini_batch(net, trainData)
            
            time = str.join(':', parts[:5])
            learning = parts[5]
            dataset = parts[6]
            losstype = parts[8]
            print('{},{},{},{},{},{},{}'.format(time, learning, dataset, netname, losstype, test_loss.item(), train_loss.item()))