import os
import argparse
import numpy as np
import pandas as pd
import time
import json
import torch

from amber.utils import run_from_ipython
from amber.backend.pytorch.session import session_scope

import pmbga
from measure_training_free_reward import get_profile, get_reward
from utils import get_data_info, get_fake_model_space, get_tmp_data
from trainer import train_arcs
from evaluate import find_top_arch, plot_history, run_evaluate
from model_builder import build_bench201_model

parser = argparse.ArgumentParser(description='train diverse tasks in biomedicine')
parser.add_argument('--dataset', required=True, type=str,
                    choices=['CIFAR100', 'ECG2017', 'NoduleMNIST3D'],
                    help='dataset string id')
parser.add_argument('--store', required=True, type=str, help='store folder to store results')
parser.add_argument('--model-space', default='bench201', type=str, help='model space string id')
parser.add_argument('--verbose', action="store_true", help="print training info")
parser.add_argument('--train-best-arc', default=1, type=int, help='the top n archs to train after searching')
parser.add_argument('--train-best-repeat', default=3, type=int, help='the number of training each arch')
parser.add_argument('--evaluate', action="store_true", help="generate evaluation figures")
##################################### Search algorithm setting #################################################
parser.add_argument('--ranking_reward', action="store_true", help='use ranking (of the raw numerical reward) as the reward for training controller.')
parser.add_argument('--buffer-size', default=5, type=int, help='number of history iterations to keep in memory')
parser.add_argument('--sample-size', default=10, type=int, help='number of sampled archs for each iteration')
parser.add_argument('--warm-up', action="store_true", help="only start updating model space after the buffer is full")
parser.add_argument('--max-iter', default=100, type=int, help='number of iteration')
parser.add_argument('--patience', default=0, type=int, help='stop search if best reward does not increase beyond patience')
##################################### Training free setting #################################################
parser.add_argument('--te-metric', nargs='+', help='training free metrics list')
parser.add_argument('--te-weight', nargs='+', help='training free metrics weights')
parser.add_argument('--te-bs', default=8, type=int, help='batch size for training free metrics')
parser.add_argument('--num_batch', type=int, default=4, help='number of batch used for training free metrics')
parser.add_argument('--repeat', default=3, type=int, help='repeat measurements')
##################################### Model setting #################################################
parser.add_argument('--width', type=int, default=16, help='initial channel width')
parser.add_argument('--n_cell', type=int, default=5, help='number of repeated cells')
##################################### Training setting #################################################
parser.add_argument('--epochs', default=100, type=int, help='training epochs')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--bs', default=256, type=int, help='batch size for training')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay')


def main():
    args = parser.parse_args()
    if args.te_metric is None:
        te_metric = ['length', 'ntk', 'synflow', 'zico']
        te_weight = [-0.5, -1, -1, 1]
    else:
        te_metric = args.te_metric
        te_weight = args.te_weights if args.te_weights else [1]*len(args.te_metric)

    # Build model space
    assert args.model_space == 'bench201'
    model_space = get_fake_model_space()

    # Read datasets.tsv - for data info
    data_info = get_data_info(dataset=args.dataset, tsv_path="datasets.tsv")

    # Prepare train data
    tmp_data = get_tmp_data(args.dataset, data_info, args.num_batch, args.te_bs)

    # Build controller
    controller = pmbga.ProbaModelBuildGeneticAlgo(
        model_space=model_space,
        buffer_type='population',
        buffer_size=args.buffer_size,
        batch_size=args.te_bs,
    )

    # Prepare log file
    timestamp = "{:}".format(time.strftime("%Y%m%d-%H%M%S"))
    job_name = f"{args.dataset}/Genetric_iter{args.max_iter}x{args.buffer_size}" + \
               f"x{args.sample_size}p{args.patience}/{timestamp}"
    save_dir = os.path.join(args.store, job_name)
    os.makedirs(save_dir, exist_ok=True)
    results = {'data': args.dataset, 'model_space': args.model_space, 'controller':[]}
    df = pd.DataFrame(columns=['iter', 'create time', 'arc', 'flops_gb', 'params_million', 'reward'] + te_metric)
    best_reward = - np.inf
    patience_cnt = 0

    # Run the iteration
    t0 = time.time()
    te_mean_all = []
    params_all = []
    for idx in range(args.max_iter):
        try:
            controller, df, te_mean_all, params_all = pmbga_iter(args, te_metric, te_weight, controller, data_info, df, 
                                                                 tmp_data=tmp_data, episode=idx, save_dir=save_dir,
                                                                 te_mean_all=te_mean_all, params_all=params_all)
            results['controller'].append(gen_results(idx, controller))
            mean_reward = df[(df['iter'] == idx) & (df['reward'] != 1)]['reward'].mean()
            if mean_reward > best_reward:
                if args.verbose: print(f"Got higher mean reward {mean_reward} than previous {best_reward}")
                best_reward = mean_reward
                patience_cnt = 0
            elif idx < args.buffer_size and args.warm_up:
                patience_cnt = 0
            else:
                patience_cnt += 1
            if args.patience>0 and patience_cnt >= args.patience:
                print("early stop due to no improvement")
                break
        except KeyboardInterrupt:
            print("User stopped training")
            break
        np.save(os.path.join(save_dir, "log.npy"), [te_mean_all, params_all])

    with open(os.path.join(save_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    df.to_csv(os.path.join(save_dir, "history.csv"), index=False)
    plot_history(df, te_metric, save_dir)

    if args.verbose: print(f"searching took {time.time() - t0} seconds..")

    # Train the best performance arch
    if args.train_best_arc > 0:
        arcs_all = find_top_arch(df, count=args.train_best_arc)['arc']
        if args.verbose: print(f"Found top {args.train_best_arc} arch: {arcs_all}")
        train_arcs(args, arcs_all, data_info, None, save_dir, repeat=args.train_best_repeat)

    if args.evaluate:
        run_evaluate(df, save_dir, args.dataset, te_metric)


def gen_results(idx, controller):

    d = {}
    for i, v in enumerate(controller.model_space):
        cnt = (v[0].Layer_attributes['operation'].obs_cnt + v[0].Layer_attributes['operation'].prior_cnt)
        d[i] = list(cnt/np.sum(cnt))

    result = {
        'iteration': idx,
        'prob': d
    }

    return result


def pmbga_iter(args, te_metric, te_weight, controller, data_info, df, tmp_data, episode, save_dir, te_mean_all, params_all):
    count = 0
    te_mean_episode = []
    action_episode = []
    while count < args.sample_size:
        # Generate sample arch
        arc = controller.get_action()
        arc_str = ''
        for i in range(len(arc)):
            arc_str += str(arc[i].Layer_attributes['operation'])

        with session_scope():
            model = build_bench201_model(arc_str, data_info, lr=0, weight_decay=0, momentum=0,
                                        width=args.width, n_cell=args.n_cell)
            model = model.cuda()
            model = model.double()

            finale_reward, all_rewards = get_reward(tmp_data, data_info, model=model, 
                                                te_metrics=te_metric, weights=te_weight, 
                                                verbose=args.verbose)
            if finale_reward == -100.:
                continue

            te_mean_all.append(finale_reward)
        
            flops, params = get_profile(tmp_data, model)
            params_all.append(params)

            model.zero_grad()
            torch.cuda.empty_cache()

        te_mean_episode.append(finale_reward) # TODO will only use for ranking purpose
        action_episode.append(arc)

        if args.verbose: print(f"Generated arch: {arc_str}, got {te_metric} = {finale_reward} | {params}")

        # Store results to controller
        if not args.ranking_reward:
            controller.store(action=arc, reward=finale_reward)
        count += 1

        # Store results to dataframe
        df.loc[len(df)] = [episode, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), arc_str, flops / 10**9 / 2, params / 10**6, finale_reward] + all_rewards

    #######################
    if args.ranking_reward:
        te_mean_episode_ranks = [sorted(te_mean_episode).index(x) for x in te_mean_episode]
        for _rank, _arc in zip(te_mean_episode_ranks, action_episode):
            controller.store(action=_arc, reward=_rank)
    #######################

    # Update the model space distribution
    if not args.warm_up or episode+2 > args.buffer_size:
        controller.train(episode=episode, working_dir=save_dir)
    
    return controller, df, te_mean_all, params_all


if __name__ == '__main__':
    if not run_from_ipython():
        main()
