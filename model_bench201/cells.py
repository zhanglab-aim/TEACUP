#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
# Modified by Tinghui Wu, 2024.02                   #
# Changes made: add 1-dimension and 3-dimension     #
#####################################################

import torch.nn as nn
from copy import deepcopy
from .cell_operations import OPS
from .utils import _init, MODE

# Cell for NAS-Bench-201
class InferCell(nn.Module):

    def __init__(self, genotype, C_in, C_out, stride):
        super(InferCell, self).__init__()

        self.layers = nn.ModuleList()
        self.node_IN = []
        self.node_IX = []
        self.genotype = deepcopy(genotype)
        for i in range(1, len(genotype)):
            node_info = genotype[i-1]
            cur_index = []
            cur_innod = []
            for (op_name, op_in) in node_info:
                if op_in == 0:
                    layer = OPS[op_name](C_in, C_out, stride, True, True)
                else:
                    layer = OPS[op_name](C_out, C_out,      1, True, True)
                _init(layer, mode=MODE)
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)
        self.nodes = len(genotype)
        self.in_dim = C_in
        self.out_dim = C_out

    def extra_repr(self):
        string = 'info :: nodes={nodes}, inC={in_dim}, outC={out_dim}'.format(
            **self.__dict__)
        laystr = []
        for i, (node_layers, node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            y = ['I{:}-L{:}'.format(_ii, _il)
                 for _il, _ii in zip(node_layers, node_innods)]
            x = '{:}<-({:})'.format(i+1, ','.join(y))
            laystr.append(x)
        return string + ', [{:}]'.format(' | '.join(laystr)) + ', {:}'.format(self.genotype.tostr())

    def forward(self, inputs):
        nodes = [inputs]
        for i, (node_layers, node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            node_feature = sum(self.layers[_il](nodes[_ii])
                               for _il, _ii in zip(node_layers, node_innods))
            nodes.append(node_feature)
        return nodes[-1]


# This module is used for NAS-Bench-201, represents a small search space with a complete DAG
class NAS201SearchCell(nn.Module):

    def __init__(self, C_in, C_out, stride, max_nodes, op_names, affine=False, track_running_stats=True):
        super(NAS201SearchCell, self).__init__()

        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                if j == 0:
                    xlists = [OPS[op_name](C_in, C_out, stride, affine, track_running_stats) for op_name in op_names]
                else:
                    xlists = [OPS[op_name](C_in, C_out, 1, affine, track_running_stats) for op_name in op_names]
                self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def extra_repr(self):
        string = 'info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
        return string

    def forward(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(sum(layer(nodes[j]) * w if w > 0.01 else 0 for layer, w in zip(self.edges[node_str], weights)))  # for pruning purpose
            nodes.append(sum(inter_nodes))
        return nodes[-1]


class MixedOp(nn.Module):

    def __init__(self, space, C, stride, affine, track_running_stats):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in space:
            op = OPS[primitive](C, C, stride, affine, track_running_stats)
            self._ops.append(op)

    def forward_darts(self, x, weights):
        return sum(w * op(x) if w > 0.01 else 0 for w, op in zip(weights, self._ops))  # for pruning purpose
