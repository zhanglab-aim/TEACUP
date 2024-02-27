from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT

import argparse
import os
import torch
import pandas as pd
import time
import warnings
from tqdm import tqdm
import json
from thop import profile
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from amber.utils import run_from_ipython
from amber.backend.pytorch.session import session_scope

from dataloader import deserilizer as get_dataloader
from rewards import deserilizer as get_reward_func
from model_store import ModelStore
from model_builder import model_builder

parser = argparse.ArgumentParser(description='train diverse tasks in biomedicine')
parser.add_argument('--dataset', required=True, type=str,
                    choices=['CIFAR100', 'ECG2017', 'NoduleMNIST3D'],
                    help='dataset string id')
parser.add_argument('--model-space', default='bench201', type=str, help='model space string id')
parser.add_argument('--resume', action='store_true', default=False, help='resume previous run by skipping existing arcs')
parser.add_argument('--db', default='store.db', type=str, help='database file to store archs and performance')
parser.add_argument('--store', required=True, type=str, help='store folder to store weights and tensorboard')
parser.add_argument('--arcs-file', required=False, type=str, help='read file for a fix set of arc')
parser.add_argument('--verbose', action="store_true", help="print training info")
##################################### Model setting #################################################
parser.add_argument('--width', type=int, default=16, help='initial channel width')
parser.add_argument('--n_cell', type=int, default=5, help='number of repeated cells')
##################################### Training setting #################################################
parser.add_argument('--epochs', default=100, type=int, help='training epochs')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay')

class CustomTernsorBoardCallback(Callback):
    # for calculating ECG reward and logging into tensorboard
    def __init__(self, valid_loader, reward_fn, step=100) -> None:
        super().__init__()
        self.step = step
        self.valid_loader = valid_loader
        self.reward_fn = reward_fn

    def on_train_batch_end(self, trainer, pl_module, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if trainer.global_step % self.step == 0:
            logger = trainer.logger
            val_reward = self.reward_fn(pl_module, self.valid_loader)[0]
            logger.log_metrics({'val_reward': val_reward}, step=trainer.global_step)
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


def main():
    enable_db_creation =  not os.path.isfile(args.db)
    model_store = ModelStore(args.db)
    if enable_db_creation:
        print("creating sql database by ./dataset.tsv; this will only run once for each db file..")
        df = pd.read_table("datasets.tsv")
        model_store.update_dataset_table(df)
        print("done!")
    
    # get archs from model_store.arch table
    if args.arcs_file:
        assert os.path.isfile(args.arcs_file), f"File path {args.arcs_file} doesn't exist!"
        print(f"Loading arcs from path {args.arcs_file}")
        if args.arcs_file.endswith(".txt"):
            with open(args.arcs_file, 'r') as f:
                arc_lines = f.readlines()
            arcs_all = [arc.strip() for arc in arc_lines][1:]
        elif args.arcs_file.endswith(".json"):
        # with open('arch_code_201.json') as json_file:
            with open(args.arcs_file, 'r') as json_file:
                arcs_all = json.load(json_file)
            # only look at first 150 for bench201 model space
            arcs_all = arcs_all[:150]
    else:
        arcs_all = model_store.get_arch_by_model_space(model_space_id=args.model_space)
    
    if len(arcs_all) == 0:
        print("initializing a new model_space and populating random 100 archs; this will only run once for each model space..")
        model_store.populate_arch_table_by_id(model_space_id=args.model_space, num_arc=100)
        arcs_all = model_store.get_arch_by_model_space(model_space_id=args.model_space)
        print("done!")

    # get this info from model_store.dataset Table
    data_info = model_store.get_dataset_info(data_id=args.dataset)
    
    # filter arcs_all if resume
    if args.resume is True:
        existing_df = model_store.to_pandas()
        existing_df = existing_df.query(f'data=="{args.dataset}" and model_space=="{args.model_space}"')
        existing_arcs = existing_df['arc'].unique()
        arcs_all_ = [arc for arc in arcs_all if arc not in existing_arcs]
        print(f"found {len(existing_arcs)} previous arcs, overlapped {len(arcs_all) - len(arcs_all_)}, needs training {len(arcs_all_)}")
        arcs_all = arcs_all_

    train_arcs(args, arcs_all, data_info, model_store, os.path.join(args.store, args.dataset))
    model_store.close()


def train_arcs(args, arcs_all, data_info, model_store, store_path, repeat=1):

    train_loader, valid_loader, test_loader = get_dataloader(args.dataset, batch_size=args.bs, data_info=data_info)

    pbar = tqdm(arcs_all, position=0, leave=True)
    for arc in pbar:
        for _ in range(repeat):
            arc_str = ''.join([str(value) for value in arc])
            timestamp = "{:}".format(time.strftime("%Y%m%d-%H%M%S"))
            job_name = "{arc_str}/{timestamp}".format(
                arc_str=arc_str,
                task=args.dataset,
                lr=args.lr,
                bs=args.bs,
                timestamp=timestamp
            )
            save_dir = "%s/%s"%(store_path, job_name)
            os.makedirs(save_dir, exist_ok=True)
            job_id = f"{args.dataset}|{timestamp}"
            pbar.set_description("Train %s "%(arc_str))
            with session_scope():
                model = model_builder(model_space=args.model_space, arc=arc, data_info=data_info,
                                      lr=args.lr, weight_decay=args.weight_decay,
                                      momentum=args.momentum, width=args.width, n_cell=args.n_cell,
                                      verbose=args.verbose)
                res = train_amber_arc(args=args, data_info=data_info, model=model, arc=arc_str, save_dir=save_dir, job_id=job_id, 
                                train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader, model_store=model_store)
    return res


def train_amber_arc(args, data_info, model, arc, save_dir, job_id, train_loader, valid_loader, test_loader, model_store=None):

    results = {'id': job_id, 'data': args.dataset, 'model_space': args.model_space, 'model_fp': os.path.realpath(save_dir),
               'reward_func': data_info['reward_func'],
               'optimizer': 'SGD', 'learning_rate': args.lr, 'batchsize': args.bs
               }

    x = torch.randn([2]+list(data_info['input_shape']))
    flops, params = profile(model, inputs=(x.double(),), verbose=False)
    arc_str = ''.join([str(value) for value in arc])
    results['arc'] = arc_str
    results['flops'] = flops
    results['params'] = params
    results['flops_gb'] = flops / 10**9 / 2
    results['params_million'] = params / 10**6
    if args.verbose:
        print("Arc:")
        print(arc_str)
        print("FLOPs = %.3fG, Params = %.3fM"%(flops / 10**9 / 2, params / 10**6))

        PID = os.getpid()
        print("PID = %d"%PID)


    t0 = time.time()
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min"
    )

    if args.dataset == 'NoduleMNIST3D':
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,
            monitor="val_loss",
            save_top_k=3,
            filename="bestmodel",
            mode="min",
        )
        callbacks = [checkpoint_callback]
    elif args.dataset == 'ECG2017':
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,
            every_n_train_steps=100,
            monitor="val_loss",
            save_top_k=3,
            filename="bestmodel",
            mode="min",
            verbose=True
        )
        reward_fn = get_reward_func(data_info['reward_func'])
        callbacks = [checkpoint_callback, CustomTernsorBoardCallback(valid_loader, reward_fn)]

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        monitor="val_loss",
        save_top_k=3,
        filename="bestmodel",
        mode="min",
        )

    callbacks = [checkpoint_callback, early_stop_callback]

    tb_logger = TensorBoardLogger(save_dir=save_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            train_loader,
            validation_data=valid_loader,
            logger=tb_logger,
            callbacks=callbacks,
            epochs=args.epochs,
            verbose=True,
        )
    if args.verbose: print(f"training took {time.time() - t0} seconds..")
    with open(os.path.join(save_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    # reload
    model = model.load_from_checkpoint(os.path.join(save_dir, "bestmodel.ckpt"), strict=False).double()#.cuda()  # pytorch-lightning will auto-use GPU
    model.eval()

    if args.verbose: print("Preparing validation data...")
    if args.verbose: print("Run validation...")

    # deserilize reward for different data/tasks
    reward_fn = get_reward_func(data_info['reward_func'])
    val_reward = reward_fn(model, valid_loader)[0]
    results["val_reward"] = val_reward
    with open(os.path.join(save_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    if args.verbose:
        print("Validation reward =", val_reward)

    if args.verbose: print("Run testing...")
    test_reward = reward_fn(model, test_loader)[0]
    results["test_reward"] = test_reward
    with open(os.path.join(save_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    if args.verbose:
        print("Test reward =", test_reward)

    if model_store:
        model_store.insert_row(row=results)
    return results


if __name__ == '__main__':
    if not run_from_ipython():
        args = parser.parse_args()
        main()
