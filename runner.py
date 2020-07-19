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

class ExptRunnerBase:
    def __init__(self, expt_prefix, net_func,
                 train_data, test_data, device=None):
        self.expt_prefix = expt_prefix
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        self.net = net_func().to(device)
        # useMultipleGpus = torch.cuda.device_count() > 1
        # if useMultipleGpus:
        #     self.net = nn.DistributedDataParallel(self.net, )
        self.expt_name = time.strftime('%m-%d-%H-%M-%S-') + expt_prefix
        self.log_folder = 'runs/' + self.expt_name
        self.checkpoint_file = self.log_folder + '/train_checkpoint.tar'
        self.writer = SummaryWriter(self.log_folder)
        self.train_mini_batch_size = 100
        self.train_loader = DataLoader(train_data, batch_size=self.train_mini_batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    
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
        total_trained = 0
        for p in self.net.parameters():
            n = p.numel()
            if p.requires_grad:
                total_trained += n
            total += n
            writeline(builder, 'Params for layer {} = {}, IsTrainable={}'.format(layer+1, n, p.requires_grad))
            layer += 1

        writeline(builder, '--------------------------------------------')
        writeline(builder, 'Total: {}'.format(total))
        writeline(builder, 'Total Trainable: {}'.format(total_trained))
        writeline(builder, '--------------------------------------------')

        with open(self.log_folder + '/network.txt', 'a') as f:
            f.write(header1.getvalue())
            f.write(str(self.net))
            f.write('\n')
            f.write(builder.getvalue())

        builder.close()
        header1.close()

    def save_matplotlib_comparison(self, rows, ip_imgs, op_imgs, filename,
                                    cmapstr='gray', shouldShow=True, printHeader=None):
        ip_imgs = ip_imgs.cpu()
        op_imgs = op_imgs.cpu()
        if printHeader is not None:
            print (printHeader)
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

        if shouldShow:
            plt.show()
        fig.savefig(self.log_folder + '/' + filename + '.png')
        self.writer.add_figure(filename, fig)

class ExptRunner(ExptRunnerBase):
    def __init__(self, expt_prefix, net_func, 
                    train_data, test_data,
                    loss_adapter_func, data_adapter_func=None, data_to_label_adapter=None, device=None):
        super(ExptRunner, self).__init__(expt_prefix, net_func, train_data, test_data, device)
        self.data_adapter_func = data_adapter_func if data_adapter_func else identity_adapter
        self.loss_adapter_func = loss_adapter_func
        self.data_to_label_adapter = data_to_label_adapter

    def run_mini_batch(self, miniBatch):
        ip_batch = self.data_adapter_func(miniBatch)
        ground_truth = self.data_to_label_adapter(miniBatch) if self.data_to_label_adapter else ip_batch
        ground_truth = ground_truth.to(self.device)
        op_batch, loss = self.loss_adapter_func(self.net, ip_batch, ground_truth)
        return ip_batch, ground_truth, op_batch, loss

    def train(self, epochs, optimizer_func=None, shouldShowReconstruction=False):
        '''
        For now, resume_previous_training can be set to True immediately after one training run
        and should use the same Trainer object with more epochs, could consider better ways in future.
        '''
        start = time.time()
        self.net.train()
        self.log_network_details()

        optimizer = optim.Adam(self.net.parameters()) if optimizer_func is None else optimizer_func(self.net)
        # For evaluating while training
        # eval_train_data = self.train_data.data[:10000]
        # eval_test_data = self.test_data.data

        builder = StringIO()
        running_loss = 0.0
        # eval_freq = 1000
        # if self.train_data.data.size()[0] < eval_freq * self.train_mini_batch_size:
        #     eval_freq = 100
        for epoch in range(epochs):
            writeline(builder, '{}: Epoch {} Begin'.format(datetime.now(), epoch))
            for i, data in enumerate(self.train_loader, 0):
                optimizer.zero_grad()

                _, _, op_batch, loss = self.run_mini_batch(data)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                # if i % eval_freq == (eval_freq-1):
                #     index = epoch * len(self.train_loader) + i
                #     writeline(builder, '{}: Eval at Index {} Begin'.format(datetime.now(), index))
                #     avg_loss = running_loss / eval_freq
                #     writeline(builder, '[%d, %5d] Average Minibatch loss: %.3f' % (epoch+1, i+1, avg_loss))
                #     self.writer.add_scalar('training_loss', avg_loss, index)
                #     running_loss = 0.0

                #     with torch.no_grad():
                #         _, _, train_out, train_loss = self.run_mini_batch(eval_train_data)
                #         writeline(builder, 'MinibatchIndex {}: Training Loss (Max 10000 rows): {}'.format(index, train_loss))

                #         test_input, test_label, test_out, test_loss = self.run_mini_batch(eval_test_data)
                #         writeline(builder, 'MinibatchIndex {}: Test Loss: {}'.format(index, test_loss))

                #     writeline(builder, '{}: Eval at Index {} End'.format(datetime.now(), index))

        writeline(builder, 'Total time taken for training {} sec.'.format(time.time() - start))

        with open(self.log_folder + '/train_log.txt', 'w') as f:
            f.write(builder.getvalue())
        builder.close()

        torch.save({
            'epoch': epochs,
            'model': self.net,
            'state_dict': self.net.state_dict(),
            'optim': optimizer,
            'optm_state_dict': optimizer.state_dict(),
            'input_adapter': self.data_adapter_func,
            'label_adapter': self.data_to_label_adapter,
            'loss_adapter': self.loss_adapter_func,
        }, self.checkpoint_file)

    def test(self, loss_adapter=None):
        self.net.eval()
        builder = StringIO()
        test_loss = 0.0
        with torch.no_grad():
            data = next(self.test_loader)
            ip_batch = self.data_adapter_func(data)
            ground_truth = self.data_to_label_adapter(data) if self.data_to_label_adapter else ip_batch
            ground_truth = ground_truth.to(self.device)
            if loss_adapter:
                op_batch, loss = loss_adapter(self.net, ip_batch, ground_truth)
                _, loss_unreduced = loss_adapter(self.net, ip_batch, ground_truth, reduction='none')
            else:
                op_batch = self.net(ip_batch)
                loss = F.l1_loss(op_batch, ground_truth)
                loss_unreduced = F.l1_loss(op_batch, ground_truth, reduction='none')

            test_loss += loss.item()
            writeline(builder, "Test Loss: {}".format(test_loss))
            loss_per_sample = torch.sum(loss_unreduced, dim=1) / loss_unreduced.size()[1]
            best_sample = torch.argmin(loss_per_sample)
            worst_sample = torch.argmax(loss_per_sample)
            writeline(builder, 'Best test sample index = {}, loss (average over dimensions) = {}'.format(best_sample, loss_per_sample[best_sample]))
            writeline(builder, 'Worst test sample index = {}, loss (average over dimensions) = {}'.format(worst_sample, loss_per_sample[worst_sample]))

            if ground_truth.size()[1] == 1024:
                batch_size = 1
                self.save_matplotlib_comparison(batch_size, 
                    demopl_v1_data_to_img(ground_truth[best_sample:best_sample+1], batch_size),
                    demopl_v1_data_to_img(op_batch[best_sample:best_sample+1], batch_size),
                    filename='best_reconstruction_{}'.format(best_sample),
                    printHeader="Final Reconstruction For Best Test Image {}".format(best_sample))
                self.save_matplotlib_comparison(batch_size, 
                    demopl_v1_data_to_img(ground_truth[worst_sample:worst_sample+1], batch_size),
                    demopl_v1_data_to_img(op_batch[worst_sample:worst_sample+1], batch_size),
                    filename='worst_reconstruction_{}'.format(worst_sample),
                    printHeader="Final Reconstruction For Worst Test Image {}".format(worst_sample))
                for n in range(5):
                    self.save_matplotlib_comparison(batch_size, 
                        demopl_v1_data_to_img(ground_truth[n:n+1], batch_size),
                        demopl_v1_data_to_img(op_batch[n], batch_size),
                        filename='final_reconstruction_{}'.format(n),
                        printHeader="Final Reconstruction For Test Image {}".format(n))

        with open(self.log_folder + '/test_log.txt', 'w') as f:
            f.write(builder.getvalue())
        builder.close()