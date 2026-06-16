# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import re
import torch
import os.path as osp
from multiprocessing import Pool
from tqdm import tqdm
from typing import Optional
from torch_geometric.datasets import *
from torch_geometric.data import Dataset
from .pyg_dataset import GraphormerPYGDataset, GraphormerPYGDatasetQM9
from .qm9 import newQM9, newHQM9
import torch.distributed as dist
from .molnet import MolNetPosDataset
from .moleculeace import MoleculeACEDataset

from torch_geometric.datasets import MoleculeNet
from torch_geometric.datasets.molecule_net import x_map, e_map
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_gz)

class MyQM7b(QM7b):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyQM9(QM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyHQM9(newHQM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyHQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyHQM9, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyZINC(ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).process()
        if dist.is_initialized():
            dist.barrier()

# class MyMoleculeNet(MoleculeNet):
import os
import re
import torch
import torch.distributed as dist
from multiprocessing import Pool
from tqdm import tqdm
from torch_geometric.datasets import MoleculeNet

class MyMoleculeNet(MoleculeNet):
    def __init__(self, root='/home/duanjw/workspace/OStars/ZJ/Transformer_bias_add_moleculenet/datasets/molecule', 
                 property=None, transform=None, pre_transform=None, pre_filter=None):
        
        self.property_name = 'esol' if property is None else property.lower()
        
        # 1. 直接调用父类初始化
        # 父类会自动执行 self._process() 和 self.data, self.slices = torch.load(...)
        # 此时磁盘上的 data.pt 必须是标准的 (data, slices) 2 元组
        super(MyMoleculeNet, self).__init__(root, name=self.property_name, 
                                            transform=transform, 
                                            pre_transform=pre_transform, 
                                            pre_filter=pre_filter)
        
        # 2. 计算回归任务的统计量
        if self.property_name in ['esol', 'lipo', 'freesolv']:
            self.mean = self.data.y.mean().item()
            self.std = self.data.y.std().item()
        else:
            self.mean, self.std = 0.0, 1.0

    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()
            return

        # 1. 读取原始数据
        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]

        process_args = []
        for line in dataset:
            line = re.sub(r'\".*\"', '', line).split(',')
            smiles = line[self.names[self.name][3]]
            ys = line[self.names[self.name][4]]
            ys = ys if isinstance(ys, list) else [ys]
            y = torch.tensor([float(y) if len(y) > 0 else float('NaN') for y in ys], 
                             dtype=torch.float).view(1, -1)
            process_args.append((smiles, y, self.name))
        
        # --- 改进点 1: 配置优化 ---
        data_list = []
        
        # --- 改进点 2: 引入分批内存管理 ---
        import gc
        
        try:
            print(f"Pre-processing {self.name} with 3D distributional features (Safe Mode)...")
            data_list = []
            skipped_3d_count = 0
            import gc

            # --- 放弃多进程，改用单进程循环以彻底规避 BrokenPipe ---
            for i, args in enumerate(tqdm(process_args)):
                try:
                    # 直接调用函数，不再通过 pool.imap
                    result = smiles_to_3d_graph(args)
                    if result is not None:
                        data_list.append(result)
                        if hasattr(result, 'is_skipped') and result.is_skipped.item():
                            skipped_3d_count += 1
                    
                    # 每 1000 个样本清理一次内存，保持低水位
                    if (i + 1) % 1000 == 0:
                        gc.collect()
                        
                except Exception as e:
                    # 这样如果某个分子崩了，你能看到是哪个 SMILES
                    print(f"\n跳过损坏样本: {args[0]} | 原因: {e}")
                    continue

            print(f"\n预处理完成，成功收集 {len(data_list)} 个样本。准备进行合并保存...")
            print(f">>> 因过于复杂跳过 3D 计算的分子数: {skipped_3d_count}")



                        
        except Exception as e:
            print(f"Multiprocessing encountered an error: {e}. Attempting to save partial results...")

        if len(data_list) == 0:
            raise ValueError("No molecules were successfully processed!")

        # 过滤
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        # 3. 使用标准 PyG 方法拼接
        print(f"Collating {len(data_list)} molecules (this may take a while for large datasets)...")
        data, slices = self.collate(data_list)

        # 4. 保存并清理内存
        print(f"Saving standard PyG dataset to {self.processed_paths[0]}...")
        torch.save((data, slices), self.processed_paths[0])
        
        # 显式释放 data_list，防止后续 get_idx_split 时内存依然处于高位
        del data_list
        gc.collect()

        if dist.is_initialized():
            dist.barrier()

    # --- 分割逻辑部分保持不变，但建议增加缓存检查 ---
    def get_idx_split(self, seed=0, split_type='random_scaffold'):
        split_path = osp.join(self.root, self.name, f"split_{split_type}_seed{seed}.pt")
        
        if osp.exists(split_path):
            return torch.load(split_path)
        
        if split_type == 'balanced_scaffold':
            split_dict = self.get_balanced_scaffold_split(seed=seed)
        elif split_type == 'random_scaffold':
            split_dict = self.get_random_scaffold_split(seed=seed)
        else:
            raise ValueError(f"Unknown split type: {split_type}")
            
        torch.save(split_dict, split_path)
        return split_dict
    
    def get_random_scaffold_split(self, seed=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        
        # 注意：对于 HIV，这里循环调用 self.get(i) 会非常慢，因为要频繁读取磁盘
        print("Generating random scaffold split...")
        smiles_list = [self.get(i).smiles for i in range(len(self))]
        
        rng = np.random.RandomState(seed)
        scaffolds = defaultdict(list)
        for ind, smiles in enumerate(smiles_list):
            scaffold = generate_scaffold(smiles, include_chirality=True)
            scaffolds[scaffold].append(ind)

        scaffold_sets = list(scaffolds.values())
        scaffold_idxes = rng.permutation(list(range(len(scaffold_sets))))

        n_total_valid = int(np.floor(frac_valid * len(self)))
        n_total_test = int(np.floor(frac_test * len(self)))

        train_idx, valid_idx, test_idx = [], [], []

        for idx in scaffold_idxes:
            scaffold_set = scaffold_sets[idx]
            if len(valid_idx) + len(scaffold_set) <= n_total_valid:
                valid_idx.extend(scaffold_set)
            elif len(test_idx) + len(scaffold_set) <= n_total_test:
                test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        return {
            "train": torch.tensor(train_idx, dtype=torch.long),
            "valid": torch.tensor(valid_idx, dtype=torch.long),
            "test": torch.tensor(test_idx, dtype=torch.long)
        }

    def get_balanced_scaffold_split(self, seed=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, balanced=True):
        """
        优化后的逻辑2：均衡骨架分割 (Balanced Scaffold Split)
        """
        import random # 确保导入
        # 优化：通过 self._data.smiles 直接获取，避免 self.get(i) 的磁盘解压开销
        print("Loading SMILES for scaffold splitting...")
        if hasattr(self._data, 'smiles') and isinstance(self._data.smiles, list):
            smiles_list = self._data.smiles
        else:
            smiles_list = [self.get(i).smiles for i in range(len(self))]
            
        train_size = frac_train * len(smiles_list)
        val_size = frac_valid * len(smiles_list)
        test_size = frac_test * len(smiles_list)

        all_scaffolds = {}
        for i, smiles in enumerate(tqdm(smiles_list, desc="Generating Scaffolds")):
            try:
                scaffold = generate_scaffold(smiles, include_chirality=True)
            except:
                continue
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)

        if balanced:
            index_sets = list(all_scaffolds.values())
            big_index_sets = []
            small_index_sets = []
            for index_set in index_sets:
                # 较大的骨架优先放入训练集，防止验证/测试集被单一骨架占据
                if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                    big_index_sets.append(index_set)
                else:
                    small_index_sets.append(index_set)
            
            # 使用局部随机状态确保可复现性
            rng = random.Random(seed)
            rng.shuffle(big_index_sets)
            rng.shuffle(small_index_sets)
            all_scaffold_sets = big_index_sets + small_index_sets
        else:
            # 按骨架大小降序排列
            all_scaffold_sets = [
                scaffold_set for (scaffold, scaffold_set) in sorted(
                    all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
            ]

        train_cutoff = frac_train * len(smiles_list)
        valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
        train_idx, valid_idx, test_idx = [], [], []

        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        return {
            "train": torch.tensor(train_idx, dtype=torch.long),
            "valid": torch.tensor(valid_idx, dtype=torch.long),
            "test": torch.tensor(test_idx, dtype=torch.long)
        }


class MyMoleculeNetPos(MolNetPosDataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNetPos, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNetPos, self).process()
        if dist.is_initialized():
            dist.barrier()


def smiles_to_3d_graph(args):
    smiles, y, name = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None

    # --- 1. 复杂度熔断检查 ---
    mol_with_h = Chem.AddHs(mol)
    num_atoms_with_h = mol_with_h.GetNumAtoms()
    
    # 设定阈值：总原子数 > 80 则标记为过载，不进行 3D 计算
    is_too_complex = num_atoms_with_h > 80

    # 2. 确定重原子索引与映射 (无论 3D 是否跳过，这步都是 2D 特征的基础)
    heavy_idx = [a.GetIdx() for a in mol_with_h.GetAtoms() if a.GetSymbol() != 'H']
    num_heavy = len(heavy_idx)
    idx_map = {old_idx: i for i, old_idx in enumerate(heavy_idx)}
    
    # --- 3. 核心改进：条件触发 3D 计算 ---
    # 初始化全 0 占位符
    pos_mean = torch.zeros((num_heavy, num_heavy), dtype=torch.float)
    pos_std = torch.zeros((num_heavy, num_heavy), dtype=torch.float)
    pos = torch.zeros((num_heavy, 3), dtype=torch.float)
    valid_3d = torch.tensor([False], dtype=torch.bool)

    if not is_too_complex:
        try:
            # 只有不复杂时才调用耗时的 3D 函数
            mu_all, sigma_all, v3d = get_distributional_3d_features(mol_with_h, n_confs=10)
            
            if v3d.item():
                # 裁剪矩阵与坐标 (严格对齐)
                pos_mean = mu_all[heavy_idx][:, heavy_idx]
                pos_std = sigma_all[heavy_idx][:, heavy_idx]
                valid_3d = v3d
                
                # 提取基准坐标
                if mol_with_h.GetNumConformers() > 0:
                    conf = mol_with_h.GetConformer(0)
                    for i, old_idx in enumerate(heavy_idx):
                        p = conf.GetAtomPosition(old_idx)
                        pos[i] = torch.tensor([p.x, p.y, p.z])
        except Exception as e:
            # 即使 3D 计算过程中意外报错，也只是跳过 3D，确保 2D 数据能存下来
            valid_3d = torch.tensor([False], dtype=torch.bool)

    # --- 4. 2D 特征提取 (这部分逻辑保持不变，确保 2D 拓扑完整) ---
    try:
        # 5. 提取原子特征
        xs = []
        for i in heavy_idx:
            atom = mol_with_h.GetAtomWithIdx(i)
            x = [
                x_map['atomic_num'].index(atom.GetAtomicNum()),
                x_map['chirality'].index(str(atom.GetChiralTag())),
                x_map['degree'].index(atom.GetTotalDegree()),
                x_map['formal_charge'].index(atom.GetFormalCharge()),
                x_map['num_hs'].index(atom.GetTotalNumHs()),
                x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()),
                x_map['hybridization'].index(str(atom.GetHybridization())),
                x_map['is_aromatic'].index(atom.GetIsAromatic()),
                x_map['is_in_ring'].index(atom.IsInRing()),
            ]
            xs.append(x)
        x_tensor = torch.tensor(xs, dtype=torch.long).view(-1, 9)

        # 6. 提取边特征
        edge_indices, edge_attrs = [], []
        for bond in mol_with_h.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if u in idx_map and v in idx_map: 
                new_u, new_v = idx_map[u], idx_map[v]
                e = [
                    e_map['bond_type'].index(str(bond.GetBondType())),
                    e_map['stereo'].index(str(bond.GetStereo())),
                    e_map['is_conjugated'].index(bond.GetIsConjugated())
                ]
                edge_indices += [[new_u, new_v], [new_v, new_u]]
                edge_attrs += [e, e]

        edge_index = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

        # 返回 Data 对象，额外增加 is_skipped 属性方便统计
        return Data(
            x=x_tensor, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            y=y, 
            pos=pos, 
            pos_mean=pos_mean.reshape(-1), 
            pos_std=pos_std.reshape(-1), 
            valid_3d=valid_3d, 
            smiles=smiles,
            is_skipped=torch.tensor([is_too_complex], dtype=torch.bool) # 标记是否被跳过
        )
    except Exception as e:
        print(f"Error in {smiles}: {e}")
        return None
    
def get_distributional_3d_features(mol, n_confs=10):
    """
    输入 RDKit Mol 对象 (需含氢)，计算全原子多构象下的距离统计量。
    """
    num_atoms = mol.GetNumAtoms()
    zeros = torch.zeros((num_atoms, num_atoms), dtype=torch.float)
    fail_result = (zeros, zeros, torch.tensor([False], dtype=torch.bool))

    try:
        # 1. 生成构象
        params = AllChem.ETKDGv3()
        params.maxAttempts = 30
        params.useSmallRingTorsions = True
        params.randomSeed = 42
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        
        if len(cids) < 2: # 至少需要2个构象才能算标准差
            return fail_result

        # 2. 力场优化 (MMFF)
        try:
            # 限制迭代次数防止 HIV 等大数据集死锁，单线程避免多进程冲突
            AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=100, numThreads=1)
        except:
            pass

        # 3. 提取坐标 [K, N, 3]
        coords_list = [mol.GetConformer(cid).GetPositions() for cid in cids]
        coords = np.array(coords_list) 
        
        # 4. 计算距离矩阵 (向量化运算加速)
        # [K, N, 1, 3] - [K, 1, N, 3] -> [K, N, N]
        delta = coords[:, :, np.newaxis, :] - coords[:, np.newaxis, :, :]
        dist_maps = np.linalg.norm(delta, axis=-1) # 比 np.sqrt(np.sum) 更快

        mu = np.mean(dist_maps, axis=0)
        sigma = np.std(dist_maps, axis=0)

        # 注意：这里不再内部过滤重原子，直接返回全原子矩阵
        return (torch.from_numpy(mu).float(), 
                torch.from_numpy(sigma).float(), 
                torch.tensor([True], dtype=torch.bool))

    except Exception:
        return fail_result
    
import random
from collections import defaultdict
import numpy as np
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold



class PYGDatasetLookupTable:
    @staticmethod
    def GetPYGDataset(dataset_spec: str, seed: int, task_idx: int = 4) -> Optional[Dataset]:
        split_result = dataset_spec.split(":")
        if len(split_result) == 2:
            name, params = split_result[0], split_result[1]
            params = params.split(",")
        elif len(split_result) == 1:
            name = dataset_spec
            params = []
        inner_dataset = None
        num_class = 1
        train_set = None
        valid_set = None
        test_set = None

        root = "dataset"
        qm9_data = False
        if name == "qm7b":
            inner_dataset = MyQM7b(root=root)
        elif name == "qm9":
            inner_dataset = MyQM9(root=root)
            qm9_data = True
        elif name == "qm9H":
            inner_dataset = MyHQM9(root=root)
            qm9_data = True
        elif name == "zinc":
            inner_dataset = MyZINC(root=root)
            train_set = MyZINC(root=root, split="train")
            valid_set = MyZINC(root=root, split="val")
            test_set = MyZINC(root=root, split="test")
        elif name.startswith("molnet"):
            property = name.split("-")[1]
            inner_dataset = MyMoleculeNetPos(property=property)
            idx_split = inner_dataset.get_idx_split(seed=seed)
            print(f"load {name} dataset with seed{seed}")
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test"]
            return GraphormerPYGDataset(
                inner_dataset,
                seed=seed,
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
            )
        elif name.startswith("moleculeace"):
            # 1. 解析任务名称
            # 期望格式: "moleculeace-chembl204_ki" -> 提取出 "chembl204_ki"
            try:
                property_name = name.split("-")[1]
            except IndexError:
                raise ValueError(f"无效的数据集名称格式: {name}。请使用 'moleculeace-任务名' 的格式。")

            # 2. 实例化我们整合后的 MoleculeACEDataset
            # 注意：root 路径应指向你存放 MoleculeACE 原始 CSV 的父目录
            inner_dataset = MoleculeACEDataset(
                root='datasets/moleculeace', 
                property_name=property_name,
                n_confs=10  # SAC 核心：生成 10 个构象进行统计校准
            )

            # 3. 获取 MoleculeACE 官方的 8:2 聚类划分索引
            # 我们在 process 阶段已经将其保存为 split_82_cluster.pt
            idx_split = inner_dataset.get_idx_split()
            
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test"]

            # 打印日志以便调试
            print(f"Successfully loaded {name}")
            print(f">>> Split Mode: MoleculeACE Official Cluster Split (8:2)")
            print(f">>> Train: {len(train_idx)} | Valid: {len(valid_idx)} | Test: {len(test_idx)}")

            # 4. 封装并返回 Transformer-M 专用的 GraphormerPYGDataset
            # 这会处理后续的 Batch 封装和特征对齐
            return GraphormerPYGDataset(
                inner_dataset,
                seed=seed,
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
            )
        elif name.startswith("molnet2d"):
            property = name.split("-")[1]
            inner_dataset = MyMoleculeNet(property=property)
            idx_split = inner_dataset.get_idx_split(seed=seed)
            print(f"load {name} dataset with seed{seed}")
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test"]
            return GraphormerPYGDataset(
                inner_dataset,
                seed=seed,
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
            )
            
        else:
            raise ValueError(f"Unknown dataset name {name} for pyg source.")
        if train_set is not None:
            return GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                ) if not qm9_data else GraphormerPYGDatasetQM9(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                )
        else:
            data_func = GraphormerPYGDataset if not qm9_data else GraphormerPYGDatasetQM9
            return (
                None
                if inner_dataset is None
                else data_func(inner_dataset, seed, task_idx=task_idx)
            )
