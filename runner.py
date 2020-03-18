import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from io import StringIO

from utils import writeline

class ExptRunner:
    def __init__(self, expt_prefix, net, data_adapter_func, loss_adapter_func):
        self.expt_prefix = expt_prefix
        self.net = net
        self.data_adapter_func = data_adapter_func
        self.loss_adapter_func = loss_adapter_func
        self.expt_name = time.strftime('%m-%d-%H-%M-%S-') + expt_prefix
        self.log_folder = 'runs/' + self.expt_name
        self.writer = SummaryWriter(self.log_folder)
        self.prev_offset = 0

    def log_network_details(self):
        header1 = StringIO()
        writeline(header1, '--------------------------------------------')
        writeline(header1, '        Network Structure')
        writeline(header1, '--------------------------------------------')

        builder = StringIO()
        writeline(builder, '--------------------------------------------')
        writeline(builder, '        Network Parameter Statistics')
        writeline(builder, '--------------------------------------------')

        total = 0
        layer = 0
        for p in self.net.parameters():
            if p.requires_grad:
                total += p.numel()
                writeline(builder, 'Params for layer {} = {}'.format(layer+1, p.numel()))
                layer += 1

        writeline(builder, '--------------------------------------------')
        writeline(builder, 'Total: {}'.format(total))
        writeline(builder, '--------------------------------------------')

        with open(self.log_folder + '/network.txt', 'w') as f:
            f.write(header1.getvalue())
            f.write(str(self.net))
            f.write('\n')
            f.write(builder.getvalue())

        builder.close()
        header1.close()


    def train(self, epochs, training_loader, resume_previous_training=False):
        '''
        For now, resume_previous_training can be set to True immediately after one training run
        and should use the same Trainer object with more epochs, could consider better ways in future.
        '''
        start = time.time()
        self.net.train()
        if not resume_previous_training:
            self.prev_offset = 0
            self.log_network_details()

        optimizer = optim.Adam(self.net.parameters(), lr=1e-4)

        builder = StringIO()
        running_loss = 0.0
        for epoch in range(epochs):
            for i, data in enumerate(training_loader, 0):
                data = data.float()
                optimizer.zero_grad()

                ip_batch = self.data_adapter_func(data)
                op_batch, loss = self.loss_adapter_func(self.net, ip_batch)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 100 == 99:
                    avg_loss = running_loss / 100
                    writeline(builder, '[%d, %5d] loss: %.3f' % (self.prev_offset+epoch+1, i+1, avg_loss))
                    index = (self.prev_offset+epoch) * len(training_loader) + i
                    self.writer.add_scalar('training_loss', avg_loss, index)
                    running_loss = 0.0

        self.prev_offset += epochs
        writeline(builder, 'Total time taken for training {} sec.'.format(time.time() - start))

        with open(self.log_folder + '/train_log.txt', 'w') as f:
            f.write(builder.getvalue())
        builder.close()

        torch.save(self.net.state_dict(), self.log_folder + '/autoenc.pth')

    def save_matplotlib_comparison(self, rows, ip_imgs, op_imgs, cmapstr='gray'):
        fig = plt.figure()
        for r in range(rows):
            ax = plt.subplot(rows, 2, r*2 + 1)
            cmap = plt.get_cmap(cmapstr)
            plt.imshow(ip_imgs[r], cmap=cmap)
            if r==0:
                ax.title.set_text('Original')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax = plt.subplot(rows, 2, r*2 + 2)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.imshow(op_imgs[r], cmap=cmap)
            if r==0:
                ax.title.set_text('Recontructed')

        plt.show()
        fig.savefig(self.log_folder + '/reconstruction.png')
        self.writer.add_figure('Reconstruction Sample', fig)


    def test(self, test_data_loader, data_to_img_func=None):
        self.net.eval()
        builder = StringIO()
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_data_loader, 0):
                data = data.float()
                ip_batch = self.data_adapter_func(data)
                op_batch, loss = self.loss_adapter_func(self.net, ip_batch)

                test_loss += loss.item()
                if data_to_img_func is not None and i == 0:
                    n = min(data.size(0), 8)
                    self.save_matplotlib_comparison(n, data_to_img_func(ip_batch[:n], n), data_to_img_func(op_batch[:n], n))

        avg_loss = test_loss / len(test_data_loader.dataset)
        writeline(builder, "Average Test Loss: {:.3f}".format(avg_loss))
        with open(self.log_folder + '/test_log.txt', 'w') as f:
            f.write(builder.getvalue())
        builder.close()
