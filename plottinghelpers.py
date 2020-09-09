import matplotlib.pyplot as plt
import utils
import torch
import os
import numpy as np

def save_recontruction_sample(ip_batch, op_batch, filePath, showimg=False):
    cmap = plt.get_cmap('gray')
    ip_batch = ip_batch.cpu()
    op_batch = op_batch.cpu()
    fig = plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.imshow(ip_batch, cmap=cmap, vmin=0, vmax=1)
    ax.title.set_text('Original')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax = plt.subplot(1, 2, 2)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.imshow(op_batch, cmap=cmap, vmin=0, vmax=1)
    ax.title.set_text('Recontructed')
    if len(filePath) > 0:
        fig.savefig(filePath)
    if showimg:
        plt.show()
    plt.close()

def generate_img_reconstructions(rootdir, device, train_data, test_data, img_res):
    for chkpt_file in utils.enumerate_files(rootdir=rootdir):
        chkpt = torch.load(chkpt_file, map_location=device)
        net = chkpt['model']
        data_adapter_func = chkpt['input_adapter']
        # data_to_label_adapter = chkpt['label_adapter']
        log_folder = os.path.dirname(chkpt_file)
        for dataset, folderName in zip([train_data, test_data], ['TrainingSet', 'TestSet']):
            results_folder = os.path.join(log_folder, folderName)
            for crit in ['good', 'bad']:
                with open(os.path.join(results_folder, '{}_samples.txt'.format(crit)), 'r') as f:
                    while True:
                        line = f.readline()
                        if len(line) == 0:
                            break
                        idx = int(line)
                        ip_batch = data_adapter_func(dataset[idx])
                        with torch.no_grad():
                            op_batch = net(ip_batch)
                        ip_batch = ip_batch.reshape(img_res, img_res)
                        op_batch = op_batch.reshape(img_res, img_res)
                        save_recontruction_sample(ip_batch, op_batch, os.path.join(results_folder, '{}_{}.png'.format(crit, idx)))

def error_interval_counts(low, high, num, errors):
    x = np.linspace(low, high, num)
    y = np.zeros(len(x))
    for i in range(0, len(x)-1):
        y[i] = np.count_nonzero(np.logical_and(np.less_equal(x[i], errors), np.less(errors, x[i+1])))
    y[-1] = np.count_nonzero(errors >= x[-1])
    assert sum(y) == len(errors)
    return x, y

def save_all_plots_policy_rollout_run(rootdir):
    error_indices = {2:'PolicyError', 3:'TrajectoryError', 4:'GoalDeviation', 5:'CombinedStateLoss'}
    upper_limits = {2:0.05, 3:0.2, 4:0.2, 5:0.1}
    metrics_file_name = 'trajectory_eval_metrics.csv'
    for  eval_folder_name in ['TestSet_Evaluations', 'SmallTrainSet_Evaluations']:
        save_all_plots(rootdir, eval_folder_name, error_indices, upper_limits, metrics_file_name)

def save_all_plots_img_auto_encoder_run(rootdir):
    error_indices = {0:'ReconstructionError'}
    upper_limits = {0:0.01}
    metrics_file_name = 'test_eval_metrics.csv'
    for eval_folder_name in ['TestSet']: # Intentionally not including 'TrainingSet' as it's too big for this scenario.
        save_all_plots(rootdir, eval_folder_name, error_indices, upper_limits, metrics_file_name)

def save_all_plots(rootdir, eval_folder_name, error_indices, upper_limits, metrics_file_name):
    for chkpt_file in utils.enumerate_files(rootdir):
        run_folder = os.path.dirname(chkpt_file)
        print ('Processing folder : {}'.format(run_folder))
        metrics = np.loadtxt(os.path.join(run_folder, eval_folder_name, metrics_file_name), delimiter=',', dtype=np.float32)
        if metrics.ndim == 1:
            metrics = metrics.reshape(-1, 1)
        for error_index, error_name in error_indices.items():
            fig = plt.figure()
            plt.boxplot(metrics[:, error_index])
            plt.ylabel(error_name)
            fig.savefig(os.path.join(run_folder, eval_folder_name, 'BoxPlot_' + error_name + '.png'))
            plt.close()
            
            fig = plt.figure()
            plt.bar(range(len(metrics)), np.sort(metrics[:, error_index]))
            plt.ylabel(error_name)
            fig.savefig(os.path.join(run_folder, eval_folder_name, 'SortedErrors_BarPlot_' + error_name + '.png'))
            plt.close()

            low, high, num = 0.0, upper_limits[error_index], 20
            x, y = error_interval_counts(low, high, num, metrics[:, error_index])
            fig = plt.figure()
            plt.bar(x, y, align='edge', width=(high-low)/num)
            plt.ylabel('Count of trajectories')
            plt.xlabel(error_name + 'Range')
            fig.savefig(os.path.join(run_folder, eval_folder_name, 'TrajectoryCounts_BarPlot_' + error_name + '.png'))
            plt.close()
            
            percentiles = np.percentile(metrics[:, error_index], [50, 95, 99], axis=0)
            np.savetxt(os.path.join(run_folder, eval_folder_name, 'Percentiles_' + error_name + '.csv'), percentiles, fmt='%2.5f', delimiter=',')
    print ('Done')

def error_metrics_list(rootdir, eval_folder_name, error_index, include=None):
    L = 0 if not include else len(include)
    errors = [0.0] * L
    run_names = [''] * L
    for chkpt_file in utils.enumerate_files(rootdir):
        run_folder = os.path.dirname(chkpt_file)
        run_name = os.path.basename(run_folder)
        if include is not None and run_name not in include:
            continue
        metrics = np.loadtxt(os.path.join(run_folder, eval_folder_name, 'trajectory_eval_metrics.csv'), delimiter=',', dtype=np.float32)
        if include is not None:
            idx = include.index(run_name)
            errors[idx] = metrics[:, error_index]
            run_names[idx] = run_name
        else:
            errors.append(metrics[:, error_index])
            run_names.append(run_name)
    return run_names, errors
        
def box_plot_across_runs(rootdir, eval_folder_name, error_index, metric_name, include=None):
    run_names, errors = error_metrics_list(rootdir, eval_folder_name, error_index, include=include)
    plt.figure()
    plt.boxplot(errors)
    plt.show()
    plt.close()
    print ('From left to right:')
    for n in run_names:
        print (n)