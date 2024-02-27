from typing import Dict
import torch
import numpy as np
import os
import math
import pickle
import pandas as pd
import random
import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Subset, DataLoader
import medmnist

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.2023, 0.1994, 0.2010]


def deserilizer(s: str, batch_size: int, data_info: Dict, augmentation: bool=True, num_workers: int=2):
    if s == 'ECG2017':
        train_loader = torch.utils.data.DataLoader(ECG2017("./data/nas-bench-360", datatype='train'), shuffle=True, batch_size=batch_size, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(ECG2017("./data/nas-bench-360", datatype='valid'), shuffle=False, batch_size=batch_size, num_workers=0)
        test_loader = torch.utils.data.DataLoader(ECG2017("./data/nas-bench-360", datatype='test'), shuffle=False, batch_size=batch_size, num_workers=0)
    elif s.startswith('CIFAR'):
        if data_info['input_shape'] and max(data_info['input_shape']) != 32:
            resize = max(data_info['input_shape'])
        else:
            resize = None
        data_dir = './data/cifar.python'
        test_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
        if resize:
            test_transform_list.append(transforms.Resize((resize, resize)))
        test_transform_list.append(transforms.Lambda(lambda img: torch.permute(img, (1, 2, 0))))
        test_transform_list.append(transforms.Lambda(lambda x: x.double()))
        test_transform = transforms.Compose(test_transform_list)

        if augmentation:
            train_transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
            if resize:
                train_transform_list.append(transforms.Resize((resize, resize)))
            train_transform_list.append(transforms.Lambda(lambda img: torch.permute(img, (1, 2, 0))))
            train_transform_list.append(transforms.Lambda(lambda x: x.double()))
        else:
            train_transform_list = list(test_transform_list)
        train_transform = transforms.Compose(train_transform_list)

        train_set = eval(s)(data_dir, train=True, transform=train_transform, download=True)
        train_set = Subset(train_set, list(range(45000)))
        val_set = Subset(eval(s)(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000))) #  check random order
        valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        test_set = eval(s)(data_dir, train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    elif s in ('PathMNIST', 'PneumoniaMNIST', 'NoduleMNIST3D', 'OrganMNIST3D'):
        if len(data_info['input_shape']) == 4:
            transform_list = [
                # transforms.ToTensor(), # this will check if the dimemsion is 2/3
                transforms.Lambda(lambda img: torch.from_numpy(img)),
                # transforms.Normalize(mean=[.5], std=[.5]),
                transforms.Lambda(lambda img: torch.permute(img, (1, 2, 3, 0))),
                transforms.Lambda(lambda x: x.double()),
            ]
        elif len(data_info['input_shape']) == 3:
            transform_list = [
                transforms.ToTensor(), # this will check if the dimemsion is 2/3
                # transforms.Normalize(mean=[.5], std=[.5]),
                transforms.Lambda(lambda img: torch.permute(img, (1, 2, 0))),
                transforms.Lambda(lambda x: x.double()),
            ]
        X_transform = transforms.Compose(transform_list)

        y_transform = None
        if data_info['loss_func'] == 'binary_crossentropy':
            y_transform = transforms.Compose([transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).type(torch.DoubleTensor))])

        DataClass = getattr(medmnist, s)
        train_set = DataClass(split='train', transform=X_transform, target_transform=y_transform, download=True)
        valid_set = DataClass(split='val', transform=X_transform, target_transform=y_transform, download=True)
        test_set = DataClass(split='test', transform=X_transform, target_transform=y_transform, download=True)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        raise Exception("unknown string id: %s" %s)
    return train_loader, valid_loader, test_loader


class ECG2017(torch.utils.data.Dataset):
    def __init__(self, dirpath, datatype: str = 'train',  window_size: int=1000, stride: int=500):
        # read pkl
        with open(os.path.join(dirpath,'challenge2017.pkl'), 'rb') as fin:
            res = pickle.load(fin)
        ## scale data
        all_data = res['data']
        for i in range(len(all_data)):
            tmp_data = all_data[i]
            tmp_std = np.std(tmp_data)
            tmp_mean = np.mean(tmp_data)
            all_data[i] = (tmp_data - tmp_mean) / tmp_std
        ## encode label
        all_label = []
        for i in res['label']:
            if i == 'N':
                all_label.append(0)
            elif i == 'A':
                all_label.append(1)
            elif i == 'O':
                all_label.append(2)
            elif i == '~':
                all_label.append(3)
        all_label = np.array(all_label)

        # split train val test
        X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_label, test_size=0.2, random_state=0)
        X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)

        # slide and cut
        #print('before: ')
        X_train, Y_train = self.slide_and_cut(X_train, Y_train, window_size=window_size, stride=stride)
        X_val, Y_val, pid_val = self.slide_and_cut(X_val, Y_val, window_size=window_size, stride=stride, output_pid=True)
        X_test, Y_test, pid_test = self.slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride,
                                                output_pid=True)
        #print('after: ')
        #print(Counter(Y_train), Counter(Y_val), Counter(Y_test))

        # shuffle train
        shuffle_pid = np.random.permutation(Y_train.shape[0])
        X_train = X_train[shuffle_pid]
        Y_train = Y_train[shuffle_pid]

        X_train = np.expand_dims(X_train, 2)
        X_val = np.expand_dims(X_val, 2)
        X_test = np.expand_dims(X_test, 2)
        Y_train = np.eye(4)[Y_train]
        Y_val = np.eye(4)[Y_val]
        Y_test = np.eye(4)[Y_test]

        # assign data
        if datatype == 'train':
            self.data, self.labels = torch.from_numpy(X_train).type(torch.DoubleTensor), torch.from_numpy(Y_train).type(torch.DoubleTensor)
        elif datatype == 'valid':
            self.data, self.labels = torch.from_numpy(X_val).type(torch.DoubleTensor), torch.from_numpy(Y_val).type(torch.DoubleTensor)
        elif datatype == 'test':
            self.data, self.labels = torch.from_numpy(X_test).type(torch.DoubleTensor), torch.from_numpy(Y_test).type(torch.DoubleTensor)
        else:
            raise ValueError(f"unknown datatype: {datatype}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()
        return self.data[idx], self.labels[idx]

    @staticmethod
    def slide_and_cut(X, Y, window_size, stride, output_pid=False, datatype=4):
        out_X = []
        out_Y = []
        out_pid = []
        n_sample = X.shape[0]
        for i in range(n_sample):
            tmp_ts = X[i]
            tmp_Y = Y[i]
            if tmp_Y == 0:
                i_stride = stride
            elif tmp_Y == 1:
                if datatype == 4:
                    i_stride = stride//6
                elif datatype == 2:
                    i_stride = stride//10
                elif datatype == 2.1:
                    i_stride = stride//7
            elif tmp_Y == 2:
                i_stride = stride//2
            elif tmp_Y == 3:
                i_stride = stride//20
            for j in range(0, len(tmp_ts)-window_size, i_stride):
                out_X.append(tmp_ts[j:j+window_size])
                out_Y.append(tmp_Y)
                out_pid.append(i)
        if output_pid:
            return np.array(out_X), np.array(out_Y), np.array(out_pid)
        else:
            return np.array(out_X), np.array(out_Y)

