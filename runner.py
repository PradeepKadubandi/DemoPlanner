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
import adapters

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
        self.train_mini_batch_size = 128
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
        self.data_adapter_func = data_adapter_func if data_adapter_func else adapters.identity_adapter
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

        builder = StringIO()
        for epoch in range(epochs):
            current_epoch_losses = []
            writeline(builder, '{}: Epoch {} Begin'.format(datetime.now(), epoch))
            for i, data in enumerate(self.train_loader, 0):
                optimizer.zero_grad()
                _, _, op_batch, loss = self.run_mini_batch(data)
                loss.backward()
                optimizer.step()
                current_epoch_losses.append(loss.item())

            avg_mbl = sum(current_epoch_losses) / len(current_epoch_losses)
            worst_mbl = max(current_epoch_losses)
            best_mbl = min(current_epoch_losses)
            self.writer.add_scalar('Average Training Loss / epoch', avg_mbl, epoch)
            self.writer.add_scalar('Worst Training Loss / epoch', worst_mbl, epoch)
            self.writer.add_scalar('Best Training Loss / epoch', best_mbl, epoch)
            writeline(builder, 'Average Training Loss for epoch {}: {}'.format(epoch, avg_mbl))
            writeline(builder, 'Worst Training Loss for epoch {}: {}'.format(epoch, worst_mbl))
            writeline(builder, 'Best Training Loss for epoch {}: {}'.format(epoch, best_mbl))

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

class ExptEvaluator:
    def __init__(self, checkpoint_file, train_data, test_data, device=None):
        self.checkpoint_file = checkpoint_file
        self.log_folder = os.path.dirname(checkpoint_file)
        self.device = device
        self.chkpt = torch.load(checkpoint_file, map_location=self.device)
        self.net = self.chkpt['model']
        self.data_adapter_func = self.chkpt['input_adapter']
        self.data_to_label_adapter = self.chkpt['label_adapter']
        self.train_data = train_data
        self.test_data = test_data
        self.train_mini_batch_size = 128

    def test(self, loss_adapter=None):
        self.net.eval()
        builder = StringIO()
        # test_loss = 0.0
        train_loader = DataLoader(self.train_data, batch_size=self.train_mini_batch_size, shuffle=False)
        test_loader = DataLoader(self.test_data, batch_size=self.train_mini_batch_size, shuffle=False)
        for loader, setName in zip([test_loader, train_loader], ['TestSet', 'TrainingSet']):
            result_folder = os.path.join(self.log_folder, setName)
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            writeline(builder, '============================================================')
            writeline(builder, '     Start Evaluation On {}'.format(setName))
            writeline(builder, '============================================================')
            good_test_samples = []
            bad_test_samples = []
            all_samples_errors = None
            with torch.no_grad():
                N = 0
                best_sample_all = None
                worst_sample_all = None
                total_loss = 0.0
                total_samples = 0 
                for i, data in enumerate(loader):
                    ip_batch = self.data_adapter_func(data)
                    ground_truth = self.data_to_label_adapter(data) if self.data_to_label_adapter else ip_batch
                    ground_truth = ground_truth.to(self.device)
                    if loss_adapter:
                        # op_batch, loss = loss_adapter(self.net, ip_batch, ground_truth)
                        _, loss_unreduced = loss_adapter(self.net, ip_batch, ground_truth, reduction='none')
                    else:
                        op_batch = self.net(ip_batch)
                        # loss = F.l1_loss(op_batch, ground_truth)
                        loss_unreduced = F.l1_loss(op_batch, ground_truth, reduction='none')

                    # test_loss += loss.item()
                    loss_per_sample = torch.sum(loss_unreduced, dim=1) / loss_unreduced.size()[1]
                    if all_samples_errors is None:
                        all_samples_errors = loss_per_sample
                    else:
                        all_samples_errors = torch.cat((all_samples_errors, loss_per_sample), dim=0)
                    best_sample = torch.argmin(loss_per_sample)
                    worst_sample = torch.argmax(loss_per_sample)
                    batch_loss = torch.sum(loss_per_sample)
                    batch_size = len(loss_per_sample)
                    N += 1
                    writeline(builder, 'Mini-batch Metrics for batch number {}'.format(N))
                    writeline(builder, "    Average Test Loss: {}".format(batch_loss.item() / batch_size))
                    writeline(builder, '    Best test sample index = {}, loss (average over dimensions) = {}'.format(best_sample, loss_per_sample[best_sample]))
                    writeline(builder, '    Worst test sample index = {}, loss (average over dimensions) = {}'.format(worst_sample, loss_per_sample[worst_sample]))
                    if best_sample_all is None or best_sample_all[1] > loss_per_sample[best_sample]:
                        best_sample_all = (best_sample, loss_per_sample[best_sample])
                        good_test_samples.append((N-1) * self.train_mini_batch_size + best_sample.item())
                        writeline(builder, 'New candidate for best sample: index = {}'.format(good_test_samples[-1]))
                    if worst_sample_all is None or worst_sample_all[1] < loss_per_sample[worst_sample]:
                        worst_sample_all = (worst_sample, loss_per_sample[worst_sample])
                        bad_test_samples.append((N-1) * self.train_mini_batch_size + worst_sample.item())
                        writeline(builder, 'New candidate for worst sample: index = {}'.format(bad_test_samples[-1]))
                    total_loss += batch_loss
                    total_samples += batch_size 

                with open(os.path.join(result_folder, 'test_eval_metrics.csv'), 'w') as f:
                    np.savetxt(f, all_samples_errors.cpu().numpy(), fmt='%5.3f', delimiter=',')

                writeline(builder, '------------------------------------------------------------')
                writeline(builder, 'Overall Metrics Across All Mini-batches')
                writeline(builder, '------------------------------------------------------------')
                writeline(builder, "Test Loss: {}".format(total_loss / total_samples))
                writeline(builder, 'Best test sample index = {}, loss (average over dimensions) = {}'.format(best_sample_all[0], best_sample_all[1]))
                writeline(builder, 'Worst test sample index = {}, loss (average over dimensions) = {}'.format(worst_sample_all[0], worst_sample_all[1]))

                with open(os.path.join(result_folder, 'good_samples.txt'), 'w') as g:
                    for v in good_test_samples:
                        g.write(str(v))
                        g.write('\n')
                with open(os.path.join(result_folder, 'bad_samples.txt'), 'w') as b:
                    for v in bad_test_samples:
                        b.write(str(v))
                        b.write('\n')

        with open(self.log_folder + '/test_log.txt', 'w') as f:
            f.write(builder.getvalue())
        builder.close()