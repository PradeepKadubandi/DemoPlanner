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
from datetime import datetime

from utils import writeline

class ExptRunner:
    def __init__(self, expt_prefix, net, 
                    train_data, test_data,
                    data_adapter_func, loss_adapter_func, data_to_label_adapter,
                    data_to_img_func=None, device=None):
        self.expt_prefix = expt_prefix
        self.net = net
        self.train_data = train_data
        self.test_data = test_data
        self.data_adapter_func = data_adapter_func
        self.loss_adapter_func = loss_adapter_func
        self.data_to_img_func = data_to_img_func
        self.data_to_label_adapter = data_to_label_adapter
        self.device = device
        self.expt_name = time.strftime('%m-%d-%H-%M-%S-') + expt_prefix
        self.log_folder = 'runs/' + self.expt_name
        self.checkpoint_file = self.log_folder + '/train_checkpoint.tar'
        self.writer = SummaryWriter(self.log_folder)
        self.train_mini_batch_size = 100
        self.train_loader = DataLoader(train_data, batch_size=self.train_mini_batch_size, shuffle=True)

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

    def run_mini_batch(self, miniBatch):
        ip_batch = self.data_adapter_func(miniBatch)
        ground_truth = self.data_to_label_adapter(miniBatch)
        ground_truth = ground_truth.to(self.device)
        op_batch, loss = self.loss_adapter_func(self.net, ip_batch, ground_truth)
        return op_batch, loss

    def train(self, epochs, shouldShowReconstruction=False):
        '''
        For now, resume_previous_training can be set to True immediately after one training run
        and should use the same Trainer object with more epochs, could consider better ways in future.
        '''
        start = time.time()
        self.net.train()
        self.log_network_details()

        optimizer = optim.Adam(self.net.parameters())
        # For evaluating while training
        eval_train_data = self.train_data.data[:10000]
        eval_test_data = self.test_data.data

        builder = StringIO()
        running_loss = 0.0
        eval_freq = 1000
        if self.train_data.data.size()[0] < eval_freq * self.train_mini_batch_size:
            eval_freq = 100
        for epoch in range(epochs):
            writeline(builder, '{}: Epoch {} Begin'.format(datetime.now(), epoch))
            for i, data in enumerate(self.train_loader, 0):
                optimizer.zero_grad()

                op_batch, loss = self.run_mini_batch(data)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % eval_freq == (eval_freq-1):
                    index = epoch * len(self.train_loader) + i
                    writeline(builder, '{}: Eval at Index {} Begin'.format(datetime.now(), index))
                    avg_loss = running_loss / eval_freq
                    writeline(builder, '[%d, %5d] Average Minibatch loss: %.3f' % (epoch+1, i+1, avg_loss))
                    self.writer.add_scalar('training_loss', avg_loss, index)
                    running_loss = 0.0

                    with torch.no_grad():
                        train_out, train_loss = self.run_mini_batch(eval_train_data)
                        writeline(builder, 'MinibatchIndex {}: Training Loss (Max 10000 rows): {}'.format(index, train_loss))

                        test_out, test_loss = self.run_mini_batch(eval_test_data)
                        writeline(builder, 'MinibatchIndex {}: Test Loss: {}'.format(index, test_loss))

                    if self.data_to_img_func is not None:
                        n = 1
                        filename='reconstruction_at_index_{}'.format(index)
                        printHeader="Reconstruction At Index {}".format(index) if shouldShowReconstruction else None
                        self.save_matplotlib_comparison(n, 
                            self.data_to_img_func(test_input[:n], n),
                            self.data_to_img_func(test_out[:n], n),
                            filename=filename,
                            printHeader=printHeader,
                            shouldShow=shouldShowReconstruction)

                    writeline(builder, '{}: Eval at Index {} End'.format(datetime.now(), index))

        writeline(builder, 'Total time taken for training {} sec.'.format(time.time() - start))

        with open(self.log_folder + '/train_log.txt', 'w') as f:
            f.write(builder.getvalue())
        builder.close()

        torch.save({
            'epoch': epochs,
            'model': self.net,
            'state_dict': self.net.state_dict(),
            'optim': optimizer,
            'input_adapter': self.data_adapter_func,
            'label_adapter': self.data_to_label_adapter,
            'loss_adapter': self.loss_adapter_func,
            'image_adapter': self.data_to_img_func,
        }, self.checkpoint_file)

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


    def test(self):
        self.net.eval()
        builder = StringIO()
        test_loss = 0.0
        with torch.no_grad():
            data = self.test_data.data
            op_batch, loss = self.run_mini_batch(data)
            test_loss += loss.item()

            if self.data_to_img_func is not None:
                batch_size = 1
                for n in range(5):
                    self.save_matplotlib_comparison(batch_size, 
                        self.data_to_img_func(ip_batch[n], batch_size),
                        self.data_to_img_func(op_batch[n], batch_size),
                        filename='final_reconstruction_{}'.format(n),
                        printHeader="Final Reconstruction For Test Image {}".format(n))

        avg_loss = test_loss / len(self.test_data.data)
        writeline(builder, "Average Test Loss: {}".format(avg_loss))
        with open(self.log_folder + '/test_log.txt', 'w') as f:
            f.write(builder.getvalue())
        builder.close()
