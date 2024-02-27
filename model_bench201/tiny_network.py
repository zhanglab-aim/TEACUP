#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
# Modified by Tinghui Wu, 2024.02                   #
# Changes made: add 1-dimension and 3-dimension     #
#####################################################
from typing import List, Text, Any
import torch
import torch.nn as nn
from .cells import InferCell
from .cell_operations import ResNetBasicblock, NAS_BENCH_201_3D, NAS_BENCH_201, NAS_BENCH_201_1D
from .genotypes import Structure as CellStructure
from .utils import _init, MODE, dag_str2code

from pdb import set_trace as bp

from typing import Tuple, List, Union
import copy
from amber.modeler.base import BaseModelBuilder
from amber import backend as F
from amber.modeler.supernet.pytorch_supernet import get_torch_actv_fn


def code2arch_str(code, dim=2):
    # 3_34_131
    # '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|'
    nodes = []
    for code_node in code.split('_'):
        _node = []
        for index, edge in enumerate(code_node):
            if dim == 3:
                _node.append(NAS_BENCH_201_3D[int(edge)] + "~" + str(index))
            elif dim == 2:
                _node.append(NAS_BENCH_201[int(edge)] + "~" + str(index))
            elif dim == 1:
                _node.append(NAS_BENCH_201_1D[int(edge)] + "~" + str(index))
        nodes.append("|" + "|".join(_node) + "|")
    return "+".join(nodes)


# https://github.com/D-X-Y/AutoDL-Projects/blob/main/exps/basic/basic-main.py
# The macro structure for architectures in NAS-Bench-201
class TinyNetwork(BaseModelBuilder):

    def __init__(self, inputs_op, output_op, model_compile_dict, C=16, N=5, dim=2, verbose=False, ):
        # super(TinyNetwork, self).__init__()
        self.inputs = inputs_op
        self.outputs = output_op
        self.model_compile_dict = model_compile_dict
        self.verbose = verbose
        self._C = C
        self._layerN = N
        self.dim = dim

    def __call__(self, model_states):
        model = self._convert(model_states, verbose=self.verbose)
        if model is not None:
            model.compile(**self.model_compile_dict)
        return model
    
    def _convert(self, arc_seq, verbose=False):

        genotype = CellStructure.str2structure(code2arch_str(arc_seq, dim=self.dim))

        C = self._C
        N = self._layerN

        layers = []
        if self.dim == 3:
            stem = nn.Sequential(
                nn.Conv3d(
                    in_channels=self.inputs.Layer_attributes["shape"][-1], 
                    out_channels=C, 
                    kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(C))            
        elif self.dim == 2:
            stem = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.inputs.Layer_attributes["shape"][-1], 
                    out_channels=C, 
                    kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(C))
        elif self.dim == 1:
            stem = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.inputs.Layer_attributes["shape"][-1],
                    out_channels=C,
                    kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(C))
        
        layers.append(('stem', stem))

        layer_channels = [C] * N + [C*2] + [C*2] * N + [C*4] + [C*4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        C_prev = C
        _init(stem, mode=MODE)

        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True, dim=self.dim)
            else:
                cell = InferCell(genotype, C_prev, C_curr, 1)
            layers.append((f"cell_{index}", cell))
            C_prev = cell.out_dim
        self._Layer = len(layers)

        if self.dim == 3:
            lastact = nn.Sequential(nn.BatchNorm3d(C_prev), 
                                    nn.ReLU(inplace=True))
            global_pooling = nn.Sequential(nn.AdaptiveAvgPool3d(1), 
                                        ViewLayer((C_prev,)))
        elif self.dim == 2:
            lastact = nn.Sequential(nn.BatchNorm2d(C_prev), 
                                    nn.ReLU(inplace=True))
            global_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                        ViewLayer((C_prev,)))
        elif self.dim == 1:
            lastact = nn.Sequential(nn.BatchNorm1d(C_prev), 
                                    nn.ReLU(inplace=True))
            global_pooling = nn.Sequential(nn.AdaptiveAvgPool1d(1), 
                                        ViewLayer((C_prev,)))
            
        _init(lastact, mode=MODE)
        layers.append((f"lastact", lastact))
        layers.append((f"global_pooling", global_pooling))

        classifier = nn.Linear(C_prev, self.outputs.Layer_attributes["units"])
        _init(classifier, mode=MODE)
        layers.append(("out", nn.Sequential(
            classifier,
            get_torch_actv_fn(self.outputs.Layer_attributes["activation"]),
            )))
        
        if verbose:
            print(layers)

        model = LightningResNet(layers=layers, model_compile_dict=self.model_compile_dict)
        return model

    def extra_repr(self):
        return ('{name}(C={_C}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))


class ViewLayer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

# # https://github.com/D-X-Y/AutoDL-Projects/blob/main/xautodl/models/shape_infers/InferTinyCellNet.py#L11
# class DynamicShapeTinyNet(nn.Module):
#     def __init__(self, channels: List[int], dag: Any, num_classes: int):
#         super(DynamicShapeTinyNet, self).__init__()
#         self._channels = channels
#         self.dag = dag_str2code(dag)
#         self.in_degree, self.out_degree = get_in_out_degree(self.dag)
#         self.genotype = CellStructure.str2structure(code2arch_str(dag))

#         if len(channels) % 3 != 2:
#             raise ValueError("invalid number of self.layers : {:}".format(len(channels)))
#         self._num_stage = N = len(channels) // 3

#         self.stem = nn.Sequential(
#             nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channels[0]),
#         )
#         _init(self.stem, mode=MODE)

#         # layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
#         layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

#         c_prev = channels[0]
#         self.cells = nn.ModuleList()
#         for index, (c_curr, reduction) in enumerate(zip(channels, layer_reductions)):
#             if reduction:
#                 cell = ResNetBasicblock(c_prev, c_curr, 2, True)
#             else:
#                 cell = InferCell(self.genotype, c_prev, c_curr, 1, degrees=[self.in_degree, self.out_degree])
#             self.cells.append(cell)
#             c_prev = cell.out_dim
#         self._num_layer = len(self.cells)

#         self.lastact = nn.Sequential(nn.BatchNorm2d(c_prev), nn.ReLU(inplace=True))
#         self.global_pooling = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(c_prev, num_classes)
#         _init(self.lastact, mode=MODE)
#         _init(self.classifier, mode=MODE)

#     def get_message(self) -> Text:
#         string = self.extra_repr()
#         for i, cell in enumerate(self.cells):
#             string += "\n {:02d}/{:02d} :: {:}".format(
#                 i, len(self.cells), cell.extra_repr()
#             )
#         return string

#     def extra_repr(self):
#         return "{name}(C={_channels}, N={_num_stage}, L={_num_layer})".format(
#             name=self.__class__.__name__, **self.__dict__
#         )

#     def forward(self, inputs):
#         feature = self.stem(inputs)
#         for i, cell in enumerate(self.cells):
#             feature = cell(feature)

#         out = self.lastact(feature)
#         out = self.global_pooling(out)
#         out = out.view(out.size(0), -1)
#         logits = self.classifier(out)

#         # return out, logits
#         return logits


class LightningResNet(F.Model):
    """LightningResNet is a subclass of pytorch_lightning.LightningModule

    It implements a basic functions of `step`, `configure_optimizers` but provides a similar
    user i/o arguments as tensorflow.keras.Model

    A module builder will add `torch.nn.Module`s to this instance, and define its forward
    pass function. Then this instance is responsible for training and evaluations.
    add module: use torch.nn.Module.add_module
    define forward pass: private __forward_tracker list
    """
    def __init__(self, layers=None, data_format='NWC', *args, **kwargs):
        super().__init__()
        self.__forward_pass_tracker = []
        self.layers = torch.nn.ModuleDict()
        self.hs = {}
        self.is_compiled = False
        self.criterion = None
        self.optimizer = None
        self.metrics = {}
        self.trainer = None
        self.data_format = data_format
        layers = layers or []
        for layer in layers:
            layer_id, operation, input_ids = layer[0], layer[1], layer[2] if len(layer)>2 else None
            self.add(layer_id=layer_id, operation=operation, input_ids=input_ids)
        self.save_hyperparameters()

    @property
    def forward_tracker(self):
        # return a read-only view
        return copy.copy(self.__forward_pass_tracker)

    def add(self, layer_id: str, operation, input_ids: Union[str, List, Tuple] = None):
        self.layers[layer_id] = operation
        self.__forward_pass_tracker.append((layer_id, input_ids))

    def forward(self, x, verbose=False):
        """Scaffold forward-pass function that follows the operations in
        the pre-set in self.__forward_pass_tracker
        """
        # permute input, if data_format has channel last
        tmp_2d = False
        if self.data_format == 'NWC':
            if x.dim() == 3:
                x = torch.permute(x, (0,2,1))
            elif x.dim() == 4:
                x = torch.permute(x, (0,3,2,1))
                tmp_2d = True
            elif x.dim() == 5:
                x = torch.permute(x, (0,4,3,2,1))
                tmp_2d = True
        # intermediate outputs, for branching models
        self.hs = {}
        # layer_id : current layer name
        # input_ids : if None,       take the output from prev layer as input
        #             if tuple/list, expect a list of layer_ids (str)
        # TODO: check output size
        for layer_id, input_ids in self.__forward_pass_tracker:
            assert layer_id in self.layers
            if verbose:
                print(layer_id)
                print([self.hs[layer_id].shape for layer_id in self.hs])
                print(input_ids)
                print(self.layers[layer_id])
            this_inputs = x if input_ids is None else self.hs[input_ids] if type(input_ids) is str else [self.hs[i] for i in input_ids]
            out = self.layers[layer_id](this_inputs)
            self.hs[layer_id] = out
            x = out
        if tmp_2d:
            out = out.squeeze(1)
        return out
