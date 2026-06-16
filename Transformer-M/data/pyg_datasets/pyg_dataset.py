# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch_geometric.data import Dataset
from sklearn.model_selection import train_test_split
from typing import List
import torch
import numpy as np

from ..wrapper import preprocess_item
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos

import copy
from functools import lru_cache
from tqdm import tqdm

class GraphormerPYGDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        seed: int = 0,
        train_idx=None,
        valid_idx=None,
        test_idx=None,
        train_set=None,
        valid_set=None,
        test_set=None,
    ):
        self.dataset = dataset
        if self.dataset is not None:
            self.num_data = len(self.dataset)
        self.seed = seed
        if train_idx is None and train_set is None:
            train_valid_idx, test_idx = train_test_split(
                np.arange(self.num_data),
                test_size=self.num_data // 10,
                random_state=seed,
            )
            train_idx, valid_idx = train_test_split(
                train_valid_idx, test_size=self.num_data // 5, random_state=seed
            )
            self.train_idx = torch.from_numpy(train_idx)
            self.valid_idx = torch.from_numpy(valid_idx)
            self.test_idx = torch.from_numpy(test_idx)
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        elif train_set is not None:
            self.num_data = len(train_set) + len(valid_set) + len(test_set)
            self.train_data = self.create_subset(train_set)
            self.valid_data = self.create_subset(valid_set)
            self.test_data = self.create_subset(test_set)
            self.train_idx = None
            self.valid_idx = None
            self.test_idx = None
        else:
            self.num_data = len(train_idx) + len(valid_idx) + len(test_idx)
            self.train_idx = train_idx
            self.valid_idx = valid_idx
            self.test_idx = test_idx
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        self.__indices__ = None

    def index_select(self, idx):
        dataset = copy.copy(self)
        dataset.dataset = self.dataset.index_select(idx)
        if isinstance(idx, torch.Tensor):
            dataset.num_data = idx.size(0)
        else:
            dataset.num_data = idx.shape[0]
        dataset.__indices__ = idx
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    def create_subset(self, subset):
        dataset = copy.copy(self)
        dataset.dataset = subset
        dataset.num_data = len(subset)
        dataset.__indices__ = None
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.dataset[idx]
            item.idx = idx
            item.y = item.y.reshape(-1)
            return preprocess_item(item)
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")

    def __len__(self):
        return self.num_data


class GraphormerPYGDatasetQM9(Dataset):
    std = 1.0
    mean = 0.0
    def __init__(
        self,
        dataset: Dataset,
        seed: int = 0,
        train_idx=None,
        valid_idx=None,
        test_idx=None,
        train_set=None,
        valid_set=None,
        test_set=None,
        task_idx=0,
        _mean=mean,
        _std=std,
        type=None,
    ):
        self._std = _std
        self._mean = _mean
        self.type = type
        self.dataset = dataset
        self.task_idx = task_idx
        self.atomref_tensor = dataset.atomref(task_idx)
        # print("mmmmmmmmmmmmmmmmmmmmmmm")
        # print(task_idx)
        # print(_mean)
        # print(_std)
        # print("0")
        if self.dataset is not None:
            self.num_data = len(self.dataset)
        self.seed = seed
        if train_idx is None and train_set is None:
            train_valid_idx, test_idx = train_test_split(
                np.arange(self.num_data),
                test_size=10831,
                random_state=seed,
            )
            train_idx, valid_idx = train_test_split(
                train_valid_idx, test_size=10000, random_state=seed
            )
            self.train_idx = torch.from_numpy(train_idx)
            self.valid_idx = torch.from_numpy(valid_idx)
            self.test_idx = torch.from_numpy(test_idx)
            self.mean = self.dataset.mean(task_idx, train_idx)
            self.std = self.dataset.std(task_idx, train_idx)
            self.train_data = self.index_select(self.train_idx, "train")
            self.valid_data = self.index_select(self.valid_idx, "valid")
            self.test_data = self.index_select(self.test_idx, "test")
        elif train_set is not None:
            self.num_data = len(train_set) + len(valid_set) + len(test_set)
            self.train_data = self.create_subset(train_set)
            self.valid_data = self.create_subset(valid_set)
            self.test_data = self.create_subset(test_set)
            self.train_idx = None
            self.valid_idx = None
            self.test_idx = None
        else:
            self.num_data = len(train_idx) + len(valid_idx) + len(test_idx)
            self.train_idx = train_idx
            self.valid_idx = valid_idx
            self.test_idx = test_idx
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        #在训练集上计算atomref修正后的mean/std
        if self.dataset is not None and self.train_idx is not None:
            train_targets = []
            if self.atomref_tensor is not  None:
                ar_table = self.atomref_tensor.view(-1)
            else:
                ar_table = None
            ##这里只是计算mean和valid值
            for i in tqdm(self.train_idx, desc="Stats"):
                data = self.dataset[i.item()]
                raw_y = data.y[0, task_idx].item()
                
                # 计算 E_ref
                if ar_table is not None:
                    atom_z = data.x[:, 0] # 原子序数
                    # 使用查表法快速求和: sum(table[z])
                    e_ref = ar_table[atom_z+1].sum().item()
                else:
                    e_ref = 0.0
                # print("999999999999999999999999999999")
                # print(self.dataset[0])
                # print(ar_table[self.dataset[0].x[:, 0]+1])
                # print(data)
                # print(ar_table)
                # print(ar_table[atom_z+1])
                # print(raw_y)
                # print(e_ref)
                # print(raw_y - e_ref)
                # if i >1:
                #     break
                train_targets.append(raw_y - e_ref)
            train_targets = torch.tensor(train_targets)
            self.mean = train_targets.mean().item()
            self.std = train_targets.std().item()
            
            print(f"Stats Ready: Mean={self.mean:.4f}, Std={self.std:.4f}")
        else:
            self.mean = 0.0 
            self.std = 1.0
        if self.train_idx is not None:
            self.train_data = self.index_select(self.train_idx, "train")
            self.valid_data = self.index_select(self.valid_idx, "valid")
            self.test_data = self.index_select(self.test_idx, "test")
        # print("mmmmmmmmmmmmmmmmmmmmmmm")
        # print(task_idx)
        # print(self.mean)
        # print(self.std)
        # print("0")
        self.__indices__ = None

    def index_select(self, idx, type=None):
        dataset = copy.copy(self)
        dataset.dataset = self.dataset.index_select(idx)
        if isinstance(idx, torch.Tensor):
            dataset.num_data = idx.size(0)
        else:
            dataset.num_data = idx.shape[0]
        dataset.__indices__ = idx
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        dataset.task_idx = self.task_idx
        dataset.mean = self.mean
        dataset.std = self.std
        dataset.atomref_tensor = self.atomref_tensor
        dataset.type = type
        return dataset

    def create_subset(self, subset):
        dataset = copy.copy(self)
        dataset.dataset = subset
        dataset.num_data = len(subset)
        dataset.__indices__ = None
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        dataset.task_idx = self.task_idx
        return dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.dataset[idx]
            item.idx = idx
            # item.y = item.y.reshape(-1)[self.task_idx].unsqueeze(0)

            raw_y = item.y.reshape(-1)[self.task_idx].unsqueeze(0)
            if self.atomref_tensor is not None:
                atom_z = item.x[:,0]
                e_ref = self.atomref_tensor[atom_z+1].sum()
            else:
                e_ref = 0.0
            y_norm = (raw_y - e_ref - self.mean) / self.std
            item.y = y_norm.view(-1)

            # y_e_ref = raw_y - e_ref
            # item.y = y_e_ref

            # print("****************")
            # print(e_ref)
            # print(raw_y - e_ref)
            # print(item.y)
            # print(self.mean)
            # print(self.std)
            item.train_mean = self.mean
            item.train_std = self.std
            item.type = self.type
            return preprocess_item(item)
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")

    def __len__(self):
        return self.num_data
