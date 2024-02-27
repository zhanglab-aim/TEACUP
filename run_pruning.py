import os, sys, time, argparse
import math
import random
from easydict import EasyDict as edict
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import json
import copy

from model_bench201 import tinysupernet
from model_bench201.cell_operations import NAS_BENCH_201, NAS_BENCH_201_1D, NAS_BENCH_201_3D
from utils import get_data_info
from dataloader import deserilizer as get_dataloader
from trainer import train_arcs
from measure_training_free_reward import get_reward

INF = 10000  # used to mark prunned operators

def time_string():
    ISOTIMEFORMAT='%Y-%m-%d %X'
    string = '[{:}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string


def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def prepare_logger(xargs):
    args = copy.deepcopy(xargs)
    from logger import Logger
    logger = Logger(args.save_dir, args.rand_seed)
    logger.log('Main Function with logger : {:}'.format(logger))
    logger.log('Arguments : -------------------------------')
    for name, value in args._get_kwargs():
        logger.log('{:16} : {:}'.format(name, value))
    logger.log("Python  Version  : {:}".format(sys.version.replace('\n', ' ')))
    # logger.log("Pillow  Version  : {:}".format(PIL.__version__))
    logger.log("PyTorch Version  : {:}".format(torch.__version__))
    # logger.log("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()))
    logger.log("CUDA available   : {:}".format(torch.cuda.is_available()))
    logger.log("CUDA GPU numbers : {:}".format(torch.cuda.device_count()))
    logger.log("CUDA_VISIBLE_DEVICES : {:}".format(os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'None'))
    return logger


def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    return model


def round_to(number, precision, eps=1e-8):
    # round to significant figure
    dtype = type(number)
    if number == 0:
        return number
    sign = number / abs(number)
    number = abs(number) + eps
    power = math.floor(math.log(number, 10)) + 1
    if dtype == int:
        return int(sign * round(number*10**(-power), precision) * 10**(power))
    else:
        return sign * round(number*10**(-power), precision) * 10**(power)


def delta2op_rankings(deltas, precision=10):
    # deltas_all: list of [delta, (idx_edge, idx_op)] for each metric_name
    # ascending sort: we want to prune op with more negative delta (metric_before_prune - metric_after_prune)
    rankings = {edge_op: [] for _, edge_op in deltas}  # dict of (edge_idx, op_idx): [rank_metric1, rank_metric2, ...]
    deltas = sorted(deltas, key=lambda tup: round_to(tup[0], precision), reverse=False)
    for idx, (delta, edge_op) in enumerate(deltas):
        # data[0]: delta; data[1]: (idx_edge, idx_op)
        assert edge_op in rankings
        if idx == 0:
            rankings[edge_op].append(idx)
        else:
            delta_prev, edge_op_prev = deltas[idx-1]
            if delta == delta_prev:
                # same te_metric_value as previous: share the same ranking
                rankings[edge_op].append(rankings[edge_op_prev][-1])
            else:
                # rankinig + 1
                rankings[edge_op].append(rankings[edge_op_prev][-1] + 1)
    rankings_list = [[k, v] for k, v in rankings.items()]  # list of (edge_idx, op_idx), [rank_metric_1, rank_metric_2, ...]
    # descending by sum of rankings: ranking the smaller the better
    rankings_sum = sorted(rankings_list, key=lambda tup: tup[1], reverse=True)  # list of (edge_idx, op_idx), [rank_metric_1, rank_metric_2, ...]
    return rankings_sum

def prune_func_rank(args, te_metric, te_weight, arch_parameters, model_config, data, data_info, precision=10, prune_number=1, dim=2):
    network_origin = tinysupernet(data_info["input_shape"], data_info["output_shape"], data_info["output_func"],
                                   model_config.C, model_config.N, model_config.max_nodes, model_config.space, dim=dim)
    init_model(network_origin, args.init)
    network_origin.set_alphas(arch_parameters)

    alpha_active = (nn.functional.softmax(arch_parameters, 1) > 0.01).float()
    prune_number = min(prune_number, alpha_active[0][0].sum()-1)  # adjust prune_number based on current remaining ops on each edge
    deltas_all = []
    history_all = {}

    # run over every pruned candidate
    pbar = tqdm(total=int(sum(alpha.sum() for alpha in alpha_active)), position=0, leave=True)
    for idx_edge in range(len(arch_parameters)):
        history_edge = {}
        # edge
        if alpha_active[idx_edge].sum() == 1:
            # only one op remaining
            continue
        for idx_op in range(len(arch_parameters[idx_edge])):
            # op
            if alpha_active[idx_edge, idx_op] > 0:
                # this edge-op not pruned yet
                # build the pruned model
                _arch_param = arch_parameters.detach().clone()
                _arch_param[idx_edge, idx_op] = -INF
                network = tinysupernet(data_info["input_shape"], data_info["output_shape"], data_info["output_func"], 
                                       model_config.C, model_config.N, model_config.max_nodes, model_config.space, dim=dim)
                network.set_alphas(_arch_param)
                repeat = args.repeat
                deltas = []
                history = []
                ###### get reward ########
                for _ in range(repeat):
                    # random reinit
                    init_model(network_origin, args.init+"_fanout" if args.init.startswith('kaiming') else args.init)  # for backward
                    # make sure network_origin and network are identical
                    for param_ori, param in zip(network_origin.parameters(), network.parameters()):
                        param.data.copy_(param_ori.data)
                    network.set_alphas(_arch_param)
                    network = network.double()
                    network_origin = network_origin.double()
                    final_reward_origin, all_reward_origin = get_reward(data, data_info, network_origin, te_metric, te_weight)
                    final_reward, all_reward = get_reward(data, data_info, network, te_metric, te_weight)
                    deltas.append(round((final_reward_origin - final_reward) / final_reward_origin, precision)) # more positive delta the more likely to be prunned
                    history.append([final_reward_origin, all_reward_origin, final_reward, all_reward])
                deltas_all.append([np.array(deltas).mean(), (idx_edge, idx_op)])
                history_edge[idx_op] = history
                network.zero_grad()
                network_origin.zero_grad()
                torch.cuda.empty_cache()
                pbar.update(1)
        history_all[idx_edge] = history_edge
    # deltas_all = [[[deltas[idx], key] for deltas, key in deltas_all] for idx in range(len(args.te_metric))]
    rankings_sum = delta2op_rankings(deltas_all, precision=precision) # list of (edge_idx, op_idx), [rank_metric_1, rank_metric_2, ...]
    edge2choice = {}  # (edge_idx): list of (edge_idx, op_idx) of length prune_number
    for edge_op, rankings in rankings_sum:
        edge_idx, op_idx = edge_op
        if edge_idx not in edge2choice:
            edge2choice[edge_idx] = [(edge_idx, op_idx)]
        elif len(edge2choice[edge_idx]) < prune_number:
            edge2choice[edge_idx].append((edge_idx, op_idx))
    choices_edges = list(edge2choice.values())
    for choices in choices_edges:
        for (edge_idx, op_idx) in choices:
            arch_parameters.data[edge_idx, op_idx] = -INF

    return arch_parameters, choices_edges, history_all


def is_single_path(network):
    arch_parameters = network.get_alphas()
    edge_active = (nn.functional.softmax(arch_parameters, 1) > 0.01).float().sum(1)
    for edge in edge_active:
        assert edge > 0
        if edge > 1:
            return False
    return True


def main(args):
    PID = os.getpid()
    assert torch.cuda.is_available(), 'CUDA is not available.'
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    prepare_seed(args.rand_seed)

    if args.te_metric is None:
        te_metric = ['length', 'ntk', 'synflow', 'zico']
        te_weight = [-0.5, -1, -1, 1]
    else:
        te_metric = args.te_metric
        te_weight = args.te_weights if args.te_weights else [1]*len(args.te_metric)

    timestamp = "{:}".format(time.strftime("%Y%m%d-%H%M%S"))

    # Read datasets.tsv - for data info
    data_info = get_data_info(dataset=args.dataset, tsv_path="datasets.tsv")

    # Prepare train data
    train_loader, _, _ = get_dataloader(args.dataset, batch_size=args.te_bs, data_info=data_info, augmentation=False)
    tmp_data = []
    while len(tmp_data) < args.num_batch:
        _tmp_data = next(iter(train_loader))
        if len(np.unique(_tmp_data[1])) == 1: continue
        if len(data_info['input_shape']) == 4: # dim=3
            _tmp_data[0] = torch.permute(_tmp_data[0], (0, 4, 3, 2, 1)).cuda()
        elif len(data_info["input_shape"]) == 3: # dim=2
            _tmp_data[0] = torch.permute(_tmp_data[0], (0, 3, 2, 1)).cuda()
        elif len((data_info["input_shape"])) == 2: # dim=1
            _tmp_data[0] = torch.permute(_tmp_data[0], (0, 2, 1)).cuda()
        _tmp_data[1] = _tmp_data[1].cuda()
        tmp_data.append(_tmp_data)

    ##### config & logging #####
    config = edict()
    config.class_num = data_info["output_shape"]
    config.xshape = data_info["input_shape"]
    config.batch_size = args.te_bs
    job_name = f"{args.dataset}/Pruning_repeat{args.repeat}-prunNum{args.prune_number}-prec{args.precision}" + \
            f"-{args.init}-batch{config['batch_size']}/{timestamp}_{args.rand_seed}"
    args.save_dir = os.path.join(args.store, job_name)
    os.makedirs(args.save_dir, exist_ok=True)
    config.save_dir = args.save_dir
    logger = prepare_logger(args)
    ###############

    logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, batch size={:}'.format(args.dataset, len(train_loader), config.batch_size))
    logger.log('||||||| {:10s} ||||||| Config={:}'.format(args.dataset, config))

    if len(data_info["input_shape"]) == 4:
        model_space = NAS_BENCH_201_3D
    elif len(data_info["input_shape"]) == 3:
        model_space = NAS_BENCH_201
    elif len(data_info["input_shape"]) == 2:
        model_space = NAS_BENCH_201_1D
    dim = len(data_info["input_shape"]) -  1

    model_config = edict({'C': args.width, 'N': args.n_cell,
                          'max_nodes': args.max_nodes,
                          'space': model_space,
                         })
    network = tinysupernet(data_info["input_shape"], data_info["output_shape"], data_info["output_func"], 
                           model_config.C, model_config.N, model_config.max_nodes, model_config.space, dim=dim)
    logger.log('model-config : {:}'.format(model_config))
    arch_parameters = network.get_alphas().detach().clone()
    arch_parameters[:, :] = 0

    # ### all params trainable (except train_bn) #########################
    # flop, param = get_model_infos(network, config.xshape)
    # logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
    logger.log('search-space [{:} ops] : {:}'.format(len(model_space), model_space))

    genotypes = {}; genotypes['arch'] = {-1: network.genotype()}

    arch_parameters_history = []
    arch_parameters_history_npy = []
    start_time = time.time()
    epoch = -1

    arch_parameters_history.append(arch_parameters.detach().clone())
    arch_parameters_history_npy.append(arch_parameters.detach().clone().cpu().numpy())
    np.save(os.path.join(args.save_dir, "arch_parameters_history.npy"), arch_parameters_history_npy)
    history_all = {}
    while not is_single_path(network):
        epoch += 1
        torch.cuda.empty_cache()
        print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, '/'.join(args.save_dir.split("/")[-6:])))

        arch_parameters, _, history = prune_func_rank(args, te_metric, te_weight, arch_parameters, model_config, tmp_data, data_info,
                                                        precision=args.precision,
                                                        prune_number=args.prune_number,
                                                        dim=dim
                                                        )
        history_all[epoch] = history
        # rebuild supernet
        network = tinysupernet(data_info["input_shape"], data_info["output_shape"], data_info["output_func"],
                               model_config.C, model_config.N, model_config.max_nodes, model_config.space, dim=dim)
        network.set_alphas(arch_parameters)

        arch_parameters_history.append(arch_parameters.detach().clone())
        arch_parameters_history_npy.append(arch_parameters.detach().clone().cpu().numpy())
        np.save(os.path.join(args.save_dir, "arch_parameters_history.npy"), arch_parameters_history_npy)
        genotypes['arch'][epoch] = network.genotype()

        logger.log('operators remaining (1s) and prunned (0s)\n{:}'.format('\n'.join([str((alpha > -INF).int()) for alpha in network.get_alphas()])))

    with open(os.path.join(args.save_dir, "history.json"), 'w', encoding='utf-8') as f:
        json.dump(history_all, f, ensure_ascii=False, indent=4)

    logger.log('<<<--->>> End: {:}'.format(network.genotype()))
    logger.log('operators remaining (1s) and prunned (0s)\n{:}'.format('\n'.join([str((alpha > -INF).int()) for alpha in network.get_alphas()])))

    end_time = time.time()
    logger.log('\n' + '-'*100)
    logger.log("Time spent: %d s"%(end_time - start_time))

    best_arc = "".join(str(alpha.argmax().item()) for alpha in network.get_alphas())
    for _ in range(3):
        train_arcs(args, [best_arc], data_info, None, args.save_dir)

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("TENAS")
    # parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--dataset', required=True, type=str, 
                        choices=['CIFAR100', 'ECG2017', 'NoduleMNIST3D'], 
                        help='dataset string id')
    parser.add_argument('--store', required=True, type=str, help='store folder to store results')
    parser.add_argument('--model-space', default='bench201', type=str, help='model space string id')
    parser.add_argument('--max_nodes', type=int, default=4, help='The maximum number of nodes.')
    # parser.add_argument('--track_running_stats', type=int, choices=[0, 1], help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
    # parser.add_argument('--save_dir', type=str, help='Folder to save checkpoints and log.')
    # parser.add_argument('--arch_nas_dataset', type=str, help='The path to load the nas-bench-201 architecture dataset (tiny-nas-benchmark).')
    parser.add_argument('--rand_seed', type=int, help='manual seed')
    parser.add_argument('--precision', type=int, default=3, help='precision for % of changes of cond(NTK) and #Regions')
    parser.add_argument('--prune_number', type=int, default=1, help='number of operator to prune on each edge per round')
    parser.add_argument('--init', default='kaiming_uniform', help='use gaussian init')
    # parser.add_argument('--super_type', type=str, default='basic',  help='type of supernet: basic or nasnet-super')
    ##################################### Training free setting #################################################
    parser.add_argument('--te-metric', nargs='+', help='training free metrics list')
    parser.add_argument('--te-weight', nargs='+', help='training free metrics weights')
    parser.add_argument('--te-bs', default=8, type=int, help='batch size for training free metrics')
    parser.add_argument('--num_batch', type=int, default=4, help='number of batch used for training free metrics')
    parser.add_argument('--repeat', default=5, type=int, help='repeat measurements')
    ##################################### Model setting #################################################
    parser.add_argument('--width', type=int, default=16, help='initial channel width')
    parser.add_argument('--n_cell', type=int, default=5, help='number of repeated cells')
    ##################################### Training setting #################################################
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=256, type=int, help='batch size for training')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--verbose', action="store_true", help="print training info")

    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    main(args)
