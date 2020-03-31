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
    def __init__(self, map_netname_net, device):
        self.map_netname_net = map_netname_net
        self.device = device

    def build_net(self, rootdir='runs'):
        for filename in glob.iglob(rootdir + '/**/*.pth', recursive=True):
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
            loss_adapter = vae_smooth_l1_loss_adapter if netname.lower().find('vae') >= 0 else smooth_l1_loss_adapter
            with torch.no_grad():
                criterion = nn.SmoothL1Loss()
                rows = min(5, len(data))
                ip_batch = demopl_v1_data_adapter(data[:rows])
                op_batch, _ = loss_adapter(net, ip_batch)
                ip_batch = ip_batch.cpu()
                op_batch = op_batch.cpu()
                for i in range(rows):
                    print('datapoint', str(i), 'test error', criterion(input=ip_batch[i], target=op_batch[i]))
                    plt.imshow(ip_batch[i].reshape(32,32), cmap=plt.get_cmap("gray"), vmin=0, vmax=1)
                    plt.show()
                    plt.imshow(op_batch[i].reshape(32,32), cmap=plt.get_cmap("gray"), vmin=0, vmax=1)
                    plt.show()

    def generate_csv_comparison_report(self, data, rootdir='runs'):
        for folder, net in self.build_net(rootdir):
            parts = folder.split('-')
            netname = parts[7]
            loss_adapter = vae_smooth_l1_loss_adapter if netname.lower().find('vae') >= 0 else smooth_l1_loss_adapter
            ip_batch = demopl_v1_data_adapter(data)
            op_batch, loss = loss_adapter(net, ip_batch)
            
            time = str.join(':', parts[:5])
            dataset = str.join('-', parts[5:7])
            losstype = parts[8]
            print ('{},{},{},{},{}'.format(time, dataset, netname, losstype, loss.item()))