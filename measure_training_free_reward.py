import argparse
from typing import Any, List
import os
import math
import numpy as np
import json
import torch
from tqdm import tqdm
from thop import profile

from amber.architect.reward import NTKReward, LRReward, LengthReward, SynFlowReward, ZiCoReward
from amber.backend.pytorch.session import session_scope

from dataloader import deserilizer as get_dataloader
from utils import get_data_info, clean_list
from model_builder import model_builder


parser = argparse.ArgumentParser(description='training-free measurements')
##################################### General setting ############################################
parser.add_argument('--dataset', required=True, type=str, 
                    choices=['CIFAR100', 'ECG2017', 'NoduleMNIST3D'],
                    help='dataset string id')
parser.add_argument('--model-space', required=True, type=str, help='model space string id, e.g. bench201, Conv1D_9_3_64, Conv2D_9_3_64')
parser.add_argument('--repeat', default=3, type=int, help='repeat measurements')
parser.add_argument('--bs', type=int, default=16, help='batch size for calculation')
parser.add_argument('--num_batch', type=int, default=8, help='batch size for calculation')
parser.add_argument('--arc-file', type=str, help='list of arcs to study')
parser.add_argument('--firstNarc', type=int, default=math.inf, help='only calculate first N arc')
parser.add_argument('--n_cell', type=int, default=5, help='number of repeated cells')
parser.add_argument('--width', type=int, default=16, help='initial channel width')
parser.add_argument('--te-metric', required=True, type=str, choices=['ntk', 'linear_region', 'length', 'synflow', 'zico'], help='dataset string id')
parser.add_argument('--exp_name', help='additional names for experiment', default=None, type=str)


def main():
    args = parser.parse_args()

    if args.arc_file.endswith(".txt"):
        with open(args.arc_file, 'r') as f:
            arc_lines = f.readlines()
        arcs = [arc.strip() for arc in arc_lines]
    elif args.arc_file.endswith(".json"):
        # with open('arch_code_201.json') as json_file:
        with open(args.arc_file, 'r') as json_file:
            arcs = json.load(json_file)
    # args.num_arc = len(arcs)
    NUM_ARC = len(arcs)

    _space_name = "%s.C%d.N%d"%(args.model_space, args.width, args.n_cell) if args.model_space == "bench201" else args.model_space
    file_name = "{te}_R{repeat}.{dataset}.{space}.Arc{arc}.BS{nb}x{bs}{exp_name}.json".format(
        te=args.te_metric,
        repeat=args.repeat,
        dataset=args.dataset,
        space=_space_name,
        arc=min(NUM_ARC, args.firstNarc),
        nb=args.num_batch,
        bs=args.bs,
        exp_name="_"+args.exp_name if args.exp_name else "",
    )
    file_name = os.path.join("results", _space_name, args.dataset, file_name)
    os.makedirs(os.path.join("results", _space_name, args.dataset), exist_ok=True)

    PID = os.getpid()
    print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, file_name))

    if args.te_metric == 'ntk':
        args.repeat += 2

    data_info = get_data_info(dataset=args.dataset, tsv_path="datasets.tsv")

    train_loader, _, _ = get_dataloader(args.dataset, batch_size=args.bs, data_info=data_info, augmentation=False)

    tmp_data = []
    # for _ in range(args.num_batch):
    while len(tmp_data) < args.num_batch:
        _tmp_data = next(iter(train_loader))
        # if len(np.unique(_tmp_data[1])) == 1: continue
        _unique = np.unique(_tmp_data[1], return_counts=True)
        if len(_unique[0]) == 1 or sum(_unique[1] / sum(_unique[1]) < 0.2): continue # TODO
        _tmp_data[0] = _tmp_data[0].cuda()
        _tmp_data[1] = _tmp_data[1].cuda()
        tmp_data.append(_tmp_data)

    results = {
        "arcs": arcs,
        args.te_metric: []
    }
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("running training-free calculations...")

    pbar = tqdm(range(min(NUM_ARC, args.firstNarc)), position=0, leave=True)
    for arc_idx in pbar:
        arc = results["arcs"][arc_idx]
        pbar.set_description("arc {arc} [{curr}/{total}]".format(
            arc=arc, curr=arc_idx+1, total=NUM_ARC,
        ))

        # arc = [int(_) for _ in arc]
        _results = []

        model = model_builder(args.model_space, arc, data_info,
                              lr=0, weight_decay=0, momentum=0,
                              width=args.width, n_cell=args.n_cell,
                              verbose=args.verbose)

        _results = measure_training_free(tmp_data, args.te_metric, data_info, model, repeat=args.repeat)

        if args.te_metric  == 'ntk':
            _results = clean_list(_results)

        results[args.te_metric].append(list(_results))

        # del model, train_loader
        torch.cuda.empty_cache()
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


def get_profile(data, model):
    with session_scope():
        # Use sampled archs to generate model
        # TODO: add other model space options

        flops, params = profile(model, inputs=(data[0][0].double(),), verbose=False)

        model.zero_grad()
        torch.cuda.empty_cache()

    return flops, params


def measure_training_free(data, te_metric, data_info, model, repeat):
    # Training free setting
    if te_metric in ["ntk", "length", "linear_region", "params"]:
        reduction = 'none'
    elif te_metric in ["synflow", "zico"]:
        reduction = 'mean'
    if data_info['loss_func'] == "binary_crossentropy":
        criterion = torch.nn.BCELoss(reduction=reduction)
    elif data_info['loss_func'] == "categorical_crossentropy":
        criterion = torch.nn.CrossEntropyLoss(reduction=reduction)

    if te_metric == "ntk":
        reward = NTKReward(criterion=criterion)
    elif te_metric == "linear_region":
        reward = LRReward()
    elif te_metric == "length":
        reward = LengthReward(criterion=criterion)
    elif te_metric == "synflow":
        reward = SynFlowReward(criterion=criterion)
    elif te_metric == "zico":
        reward = ZiCoReward(criterion=criterion)
    elif te_metric == "flops":
        flops, _ = get_profile(data, model)
        return [flops]
    elif te_metric == "params":
        _, params = get_profile(data, model)
        return [params]

    _results = []
    for _ in range(repeat):
        with session_scope():
            # Calculate training free metrics
            _results.append(reward(model, data)[0])
            if te_metric == "ntk":
                _results[-1] = - _results[-1]

            model.zero_grad()
            torch.cuda.empty_cache()
    return _results


def get_reward(tmp_data, data_info, model, te_metrics: List, weights: List, verbose=False):
    all_rewards = []

    for te_metric in te_metrics:
        if te_metric == 'ntk':
            results = measure_training_free(tmp_data, te_metric, data_info, model, repeat=5)
            if results[0] == -1.0:
                if verbose: print(f'Got {te_metric} == -1, will skip this')
                return -100.0, [0] * len(te_metrics)
            results = clean_list(results)
        else:
            results = measure_training_free(tmp_data, te_metric, data_info, model, repeat=3)
            if results[0] == 0.0:
                if verbose: print(f'Got {te_metric} == 0, will skip this')
                return -100.0, [0] * len(te_metrics)

        results = [np.log10(np.abs(item)) for item in results]
        te_mean = sum(results)/len(results)
        all_rewards.append(te_mean)

    final_reward = 0
    for te_mean, weight in zip(all_rewards, weights):
        final_reward += weight * te_mean

    return final_reward, all_rewards

if __name__ == '__main__':
    main()
