import random
import torch
import torch.nn as nn
from copy import deepcopy
from .cell_operations import ResNetBasicblock
from .cells     import NAS201SearchCell as SearchCell
from .genotypes        import Structure
import torch.nn.functional as F
from pdb import set_trace as bp

from amber.modeler.supernet.pytorch_supernet import get_torch_actv_fn

def cal_entropy(logit: torch.Tensor, dim=-1) -> torch.Tensor:
    """
    :param logit: An unnormalized vector.
    :param dim: ~
    :return: entropy
    """
    prob = F.softmax(logit, dim=dim)
    log_prob = F.log_softmax(logit, dim=dim)

    entropy = -(log_prob * prob).sum(-1, keepdim=False)

    return entropy


class TinySuperNet(nn.Module):

    def __init__(self, input_shape, output_shape, output_func, C, N, max_nodes, search_space, dim=2, affine=True, track_running_stats=True):
        super(TinySuperNet, self).__init__()
        self._C        = C
        self._layerN   = N  # number of stacked cell at each stage
        self.max_nodes = max_nodes

        if dim == 3:
            self.stem = nn.Sequential(nn.Conv3d(in_channels=input_shape[-1], 
                                                out_channels=C, 
                                                kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm3d(C))
        elif dim == 2:
            self.stem = nn.Sequential(nn.Conv2d(in_channels=input_shape[-1],
                                                out_channels=C, 
                                                kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(C))
        elif dim == 1:
            self.stem = nn.Sequential(nn.Conv1d(in_channels=input_shape[-1], 
                                                out_channels=C, 
                                                kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm1d(C))            

        layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            # if depth > 0 and index >= depth: break
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, dim=dim)
            else:
                cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
                if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
                else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
            self.cells.append( cell )
            C_prev = cell.out_dim
        self.op_names   = deepcopy( search_space )
        self._Layer     = len(self.cells)
        self.edge2index = edge2index

        if dim == 3:
            self.lastact = nn.Sequential(nn.BatchNorm3d(C_prev), 
                                         nn.ReLU(inplace=True))
            self.global_pooling = nn.AdaptiveAvgPool3d(1)
        elif dim == 2:
            self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), 
                                         nn.ReLU(inplace=True))
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
        elif dim == 1:
            self.lastact = nn.Sequential(nn.BatchNorm1d(C_prev), 
                                         nn.ReLU(inplace=True))
            self.global_pooling = nn.AdaptiveAvgPool1d(1)
            
        classifier = nn.Linear(C_prev, output_shape)
        self.classifier = nn.Sequential(classifier,
                                        get_torch_actv_fn(output_func))
        self.arch_parameters = nn.Parameter( 1e-3*torch.randn(num_edge, len(search_space)) )

    def entropy(self, mean=True):
        if mean:
            return cal_entropy(self.arch_parameters, -1).mean().view(-1)
        else:
            return cal_entropy(self.arch_parameters, -1)

    def get_weights(self):
        xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
        xlist += list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
        xlist += list( self.classifier.parameters() )
        return xlist

    def get_alphas(self):
        return self.arch_parameters

    def set_alphas(self, arch_parameters):
        self.arch_parameters.data.copy_(arch_parameters.data)

    def show_alphas(self):
        with torch.no_grad():
            return 'arch-parameters :\n{:}'.format( nn.functional.softmax(self.arch_parameters, dim=-1).cpu() )

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

    def genotype(self, get_random=False, hardwts=None):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    if hardwts is not None:
                        weights = hardwts[ self.edge2index[node_str] ]
                        op_name = self.op_names[ weights.argmax().item() ]
                    elif get_random:
                        op_name = random.choice(self.op_names)
                    else:
                        weights = self.arch_parameters[ self.edge2index[node_str] ]
                        op_name = self.op_names[ weights.argmax().item() ]
                xlist.append((op_name, j))
            genotypes.append( tuple(xlist) )
        return Structure( genotypes )

    def forward(self, inputs, return_features=False):
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
        features_all = []
        feature = self.stem(inputs)
        features_all.append(feature.detach())
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell(feature, alphas)
            else:
                feature = cell(feature)
            features_all.append(feature.detach())

        out = self.lastact(feature)
        out = self.global_pooling( out )
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        logits = logits.squeeze(1)

        return logits
