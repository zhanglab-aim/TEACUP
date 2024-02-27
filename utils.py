import pandas as pd
from typing import List
import numpy as np

from amber.architect import ModelSpace, Operation

from pmbga.bayes_prob import Categorical
from dataloader import deserilizer as get_dataloader

def get_data_info(dataset, tsv_path="datasets.tsv"):
    DATASET_INFO = pd.read_table(tsv_path).set_index('id').to_dict('index')
    data_info = DATASET_INFO[dataset]
    data_info['input_shape'] = eval(data_info['input_shape']) if type(data_info['input_shape']) is str else data_info['input_shape']
    data_info['output_shape'] = eval(data_info['output_shape']) if type(data_info['output_shape']) is str else data_info['output_shape']
    return data_info


def get_fake_model_space(num_op=5):
    ms = ModelSpace.from_dict([
         # 6 edges, each has 5 operatoers
        [Operation('bench201_op', operation=Categorical(choices=range(num_op), prior_cnt=1))],
        [Operation('bench201_op', operation=Categorical(choices=range(num_op), prior_cnt=1))],
        [Operation('bench201_op', operation=Categorical(choices=range(num_op), prior_cnt=1))],
        [Operation('bench201_op', operation=Categorical(choices=range(num_op), prior_cnt=1))],
        [Operation('bench201_op', operation=Categorical(choices=range(num_op), prior_cnt=1))],
        [Operation('bench201_op', operation=Categorical(choices=range(num_op), prior_cnt=1))],
    ])
    return ms


def get_fake_model_space_rl(num_op=5):
    ms = ModelSpace.from_dict([
        *[[Operation('bench201_op', operation=i) for i in range(num_op)]]*6 # 6 edges, each has 5 operatoers
    ])
    return ms


def clean_list(results: List[float]) -> List[float]:
    results.remove(max(results))
    results.remove(min(results))
    return results


def get_tmp_data(task: str, data_info: dict, num_batch: int, bs: int) -> List:
    train_loader, _, _ = get_dataloader(task, batch_size=bs, data_info=data_info, augmentation=False)
    tmp_data = []
    while len(tmp_data) < num_batch:
        _tmp_data = next(iter(train_loader))
        # if len(np.unique(_tmp_data[1])) == 1: continue
        _unique = np.unique(_tmp_data[1], return_counts=True)
        if len(_unique[0]) == 1 or sum(_unique[1] / sum(_unique[1]) < 0.2): continue # TODO
        _tmp_data[0] = _tmp_data[0].cuda()
        _tmp_data[1] = _tmp_data[1].cuda()
        tmp_data.append(_tmp_data)
    return tmp_data