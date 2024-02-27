import os
import numpy as np
import pandas as pd
import argparse
import json
from glob import glob
from matplotlib import pyplot as plt
from typing import List

from amber.utils import run_from_ipython

from trainer import train_arcs
from measure_training_free_reward import measure_training_free
from utils import get_data_info, get_tmp_data, clean_list


parser = argparse.ArgumentParser(description='train diverse tasks in biomedicine')
parser.add_argument('--save-dir', required=True, type=str, help='path to the search algorithm results')
parser.add_argument('--dataset', required=True, type=str, choices=['CIFAR100', 'ECG2017', 'NoduleMNIST3D'], help='dataset string id')
parser.add_argument('--model-space', required=True, type=str, help='model space string id')
parser.add_argument('--retrain-dir', default=None, type=str, help='store folder for new training results')
parser.add_argument('--te-metric', nargs='+', help='training free metrics list')
parser.add_argument('--verbose', action="store_true", help="print training info")
##################################### Model setting #################################################
parser.add_argument('--width', type=int, default=16, help='initial channel width')
parser.add_argument('--n_cell', type=int, default=5, help='number of repeated cells')
##################################### Training setting #################################################
parser.add_argument('--epochs', default=100, type=int, help='training epochs')
# parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate') Conv2D
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--bs', default=256, type=int, help='batch size for training')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
# parser.add_argument('--weight_decay', default=1e-7, type=float, help='weight decay') Conv2D
parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay')


def run_evaluate(history: pd.DataFrame, save_dir: str, dataset: str, te_metric: List[str]):
    df_best = find_top_arch(history, count=3)
    df_best = add_train_results(df_best, save_dir)

    benchmark_path = '/common/zhangz2lab/shared/projects/amber_lite/benchmark.csv'
    benchmark = pd.read_csv(benchmark_path, header=[0,1], index_col=0)
    plot_te_scatter(dataset, te_metric, benchmark, df_best, save_dir)
    plot_compare_box(dataset, benchmark, df_best, save_dir)


def add_train_results(df_best: pd.DataFrame, result_dir: str, lr: float=0.1) -> pd.DataFrame:
    """fetch training results to best arch dataframe

    Parameters
    ----------
    df_best : pd.DataFrame
        dataframe with the best arch(es)
    result_dir : str
        path that store the training results
    """

    val_rewards = []
    test_rewards = []
    for arc in df_best['arc']:
        valid_best, test_best = 0, 0
        result_paths = glob(os.path.join(result_dir, arc, '*', 'results.json'))
        for result_path in result_paths:
            with open(result_path, 'r') as f:
                data = json.load(f)
                if not data['learning_rate'] == lr:
                    continue
                if data['test_reward'] > test_best:
                    valid_best = data['val_reward']
                    test_best = data['test_reward']
        val_rewards.append(valid_best)
        test_rewards.append(test_best)

    df_best['val_reward'] = val_rewards
    df_best['test_reward'] = test_rewards
    return df_best

       
def plot_history(history: pd.DataFrame, metrics: List[str], save_path: str = None):
    history = history[history['reward'] != 1]
    df = history.groupby('iter').agg('mean', numeric_only=True)
    plt.plot(df[[metric for metric in metrics]], 'o')
    plt.legend(metrics)
    plt.xlabel('episode')
    plt.ylabel('log value')
    if save_path:
        plt.savefig(os.path.join(save_path, 'reward_plot.png'))


def get_te_mean(te_metric: str, data_info: dict, arc_str: str, tmp_data) -> float:
    if te_metric == 'ntk':
        results = measure_training_free(tmp_data, te_metric, data_info, arc_str, repeat=5, width=16, n_cell=5)
        results = clean_list(results)
    else:
        results = measure_training_free(tmp_data, te_metric, data_info, arc_str, repeat=3, width=16, n_cell=5)

    results = [np.log10(np.abs(item)) for item in results]
    te_mean = sum(results)/len(results)

    return te_mean
    

def plot_te_scatter(task: str, df: pd.DataFrame, df_best: pd.DataFrame, metrics: List[str]=[], 
                    save_path: str=None, mode: str='both', method: str=None, verbose: bool=False):
    te_dict = {
        'CIFAR100': {'length': (8,32), 'ntk': (8,32), 'synflow': (8,32), 'zico': (8,32)},
        'MHIST': {'length': (1,32), 'ntk': (1,32), 'synflow': (10,32), 'zico': (10,32)},
        'ECG2017': {'length': (10,32), 'ntk': (10,32), 'synflow': (10,32), 'zico': (10,32)},
        'NoduleMNIST3D': {'length': (10,32), 'ntk': (1,32), 'synflow': (10,32), 'zico': (10,32)}
    }
    fig, ax = plt.subplots(3,2, figsize=(12,18))
    if len(metrics) == 0:
        metrics = ['params_million', 'flops_gb', 'length', 'ntk', 'synflow', 'zico']
    else:
        metrics = ['params_million', 'flops_gb'] + metrics
    data_info = get_data_info(dataset=task, tsv_path="datasets.tsv")
    for idx, metric in enumerate(['params_million', 'flops_gb', 'length', 'ntk', 'synflow', 'zico']):
        ax[idx // 2, idx % 2].scatter(x=df[task, metric], y=df[task, 'test_reward'], label='benchmark')
        if metric in metrics:
            if metric not in ('params_million', 'flops_gb'):
                if mode in ('both', 'original'):
                    ax[idx // 2, idx % 2].scatter(x=np.power(10, df_best[metric]), y=df_best['test_reward'], label='original')
                if mode in ('both', 'correction'):
                    # re-calculate training free metrics
                    num_batch, bs = te_dict[task][metric]
                    tmp_data = get_tmp_data(task, data_info, num_batch, bs)
                    te_list = []
                    for arc in df_best['arc']:
                        te_list.append(get_te_mean(metric, data_info, arc, tmp_data))
                        if verbose: print(f"{task} new te metric {metric} from {arc}: {te_list} vs old {list(df_best[metric])}")
                    ax[idx // 2, idx % 2].scatter(x=np.power(10, te_list), y=df_best['test_reward'], label='correction')
            else:
                ax[idx // 2, idx % 2].scatter(x=df_best[metric], y=df_best['test_reward'], label='original')
        ax[idx // 2, idx % 2].set_title(f"{metric}") 
        ax[idx // 2, idx % 2].legend()
        if metric in ('ntk', 'length'):
            ax[idx//2, idx%2].set_xscale('log')
    if save_path:
        plt.savefig(os.path.join(save_path, f'{task}_{method}_{mode}_te_scatter.png'))


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)


def plot_compare_box(task: str, df: pd.DataFrame, df_best: pd.DataFrame, save_path: str = None):
    tasks = [task]
    plt.figure()
    for idx, task in enumerate(tasks):
        val = plt.boxplot(df[task, 'val_reward'], positions=[idx*4-1.2], widths=0.6)
        set_box_color(val, 'limegreen')
        test = plt.boxplot(df[task, 'test_reward'], positions=[idx*4+0.4], widths=0.6)
        set_box_color(test, 'blue')

    for idx, task in enumerate(tasks):
        te_val = plt.boxplot(df_best['val_reward'], positions=[idx*4-0.4], widths=0.6)
        set_box_color(te_val, 'darkgreen')
        te_test = plt.boxplot(df_best['test_reward'], positions=[idx*4+1.2], widths=0.6)
        set_box_color(te_test, 'purple')
    plt.legend([val["boxes"][0], te_val["boxes"][0], test["boxes"][0], te_test["boxes"][0]], ['val', 'te_val', 'test', 'te_test'], loc='lower right')
    plt.xticks(range(0, len(tasks)*4, 4), tasks)
    plt.grid()
    if save_path:
        plt.savefig(os.path.join(save_path, 'box_plot.png'))


def find_top_arch(df: pd.DataFrame, count: int = 5) -> pd.DataFrame:
    df = df[df['reward'] != 1]
    df = df.sort_values(by=['reward'], ascending=False).reset_index()

    picked_idx = []
    idx = 0
    while len(picked_idx) < count:
        if not df.loc[idx, 'arc'] in list(df.iloc[picked_idx]['arc']):
            picked_idx.append(idx)
        idx += 1
    return df.iloc[picked_idx]

