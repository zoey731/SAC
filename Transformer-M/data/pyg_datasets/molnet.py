from typing import List
import torch
import numpy as np
import os.path as osp
import pickle
import gc
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from multiprocessing import Pool
from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector)
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from typing import List, Union, Dict, Set
import random
from ogb.utils import smiles2graph
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})

MOLNET_TASKS = {
    "esol": ["measured log solubility in mols per litre"],
    "freesolv": ["expt"],
    "lipophilicity": ["exp"],
    "bace": ["Class"],
    "bbbp": ["p_np"],
    "clintox": ["FDA_APPROVED", "CT_TOX"],
    "tox21": [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ],
    "sider": ["Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", ...], # 依此类推
}


class MolNetPosDataset(InMemoryDataset):
    def __init__(self, root='/home/zhaoqc/Transformer_bias_add_moleculenet/datasets/moleculenet', smiles2graph=smiles2graph, transform=None, pre_transform=None, property=None): 
        '''
        '''
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.property = 'esol' if property is None else property
        self.folder = self.property
        super().__init__(osp.join(root, self.folder), transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.property.lower() in ["freesolv", "esol", "lipo"]:
            # 注意：如果 y 包含 NaN，需使用 nanmean/nanstd 或忽略 NaN
            y_all = self.data.y
            self._y_mean = float(torch.mean(y_all[~torch.isnan(y_all)]))
            self._y_std = float(torch.std(y_all[~torch.isnan(y_all)]))
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
        print("标签值平均值为：", self._y_mean)

    @property
    def y_mean(self):
        return self._y_mean

    @property
    def y_std(self):
        return self._y_std
    
    def download(self):
        # 移除 super().download()
        return
    
    def get(self, idx):
        data = super().get(idx)
        # 强制把全局统计量挂在每个 Data 对象上，这样 Collator 必能拿到
        data.y_mean = self._y_mean
        data.y_std = self._y_std
        return data

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['pos_mean', 'pos_std']:
            return 0 
        return super().__cat_dim__(key, value, *args, **kwargs)

    @property
    def raw_file_names(self):
        return self.property + '.pkl'


    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'
    

    def mean(self, idxs):
        if self.property in ["freesolv", "esol", "lipo"]:
            y = torch.cat([self.get(i).y for i in idxs], dim=0)
            print("标签值平均值为：", y[:].mean().item())
            return y[:].mean().item()
        else:
            return 0.0

    def std(self, idxs):
        if self.property in ["freesolv", "esol", "lipo"]:
            y = torch.cat([self.get(i).y for i in idxs], dim=0)
            print("标签值标准差为：", y[:].std().item())
            return y[:].std().item()
        else:
            return 1.0


    def process(self):
        if osp.exists(self.processed_paths[0]):
            print(f"检测到已存在的处理文件: {self.processed_paths[0]}, 跳过预处理。")
            # 如果你希望即使跳过 process 也要确保 split 文件存在，可以在这里检查 split 文件
            return
        
        with open(osp.join(self.root, self.raw_file_names), "rb") as f:
            data_df = pickle.load(f)
        conf_list = data_df['conf'].tolist()
        smiles_list = data_df['smiles'].tolist()
        # 获取标签逻辑 (根据你的任务修改)
        target_cols = MOLNET_TASKS[self.property.lower()]
        y_values = data_df[target_cols].values

        data_list = []
        skipped_count = 0
        print(f"正在处理 {self.property} 数据集...")
        for i, mol in enumerate(tqdm(conf_list)):
            if mol is None:
                continue
            
            try:
                # 1. 统一去氢，确保 2D 图结构和 3D 裁剪基准一致
                # 如果 RemoveHs 报错，说明 mol 对象有问题
                mol = Chem.RemoveHs(mol)
                
                # 2. 生成基础图
                graph = mol2graph(mol)
                if graph is None: continue
                
                data = Data()
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                data.pos = torch.from_numpy(graph['position']).to(torch.float32)
                data.y = torch.from_numpy(y_values[i].astype(np.float32)).view(1, -1)
                data.smiles = smiles_list[i]
                    
                # 3. 叠加 3D 统计量 (内部会自动处理 AddHs 算 3D 和重原子切片)
                data = self.augment_3d_statistics(data, mol)
                
                data_list.append(data)
                
            except Exception as e:
                skipped_count += 1
            # 只有前几个错误打印详细信息，避免刷屏
                if skipped_count < 10:
                    print(f"\n跳过分子 index {i}: {e}")
                continue
        
        print(f"处理完成。总数: {len(conf_list)}, 成功: {len(data_list)}, 踢掉: {skipped_count}")
        if len(data_list) == 0:
            raise RuntimeError("❌ 严重错误: data_list 为空！请检查 mol 对象和 mol2graph 逻辑。")
        
        data, slices = self.collate(data_list)

        # print("生成三维失败的分子数量 (可能过于复杂): ", skipped_3d)
        print('Saving processed data of ' + self.property)
        torch.save((data, slices), self.processed_paths[0])

        split = {}
        # print(f"suc/fail: {suc}/{fail}")

        seed = 42
        # scaffold split
        train, valid, test = random_scaffold_split(conf_list, balanced=False, seed=seed)  ##注意这里的分割是设置的balanced split ，但是不是按照骨架从大到小的balanced_split，而是按分子数分大骨架和小骨架，组内打乱
        split['train'] = train
        split['valid'] = valid
        split['test'] = test
        print(f"Saving new split idx with seed{seed}")
        torch.save(split, osp.join(self.root, f"split_random_scaffold_seed{seed}.pt"))


    def get_idx_split(self, seed):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, f'split_random_scaffold_seed{seed}.pt')))
        for k, v in split_dict.items():
            split_dict[k] = torch.tensor(v)
        return split_dict

    def augment_3d_statistics(self, data, mol):
        # 1. 临时加氢算 3D
        mol_with_h = Chem.AddHs(mol)
        
        # 2. 算 3D
        mu_all, sigma_all, v3d = get_distributional_3d_features(mol_with_h, n_confs=10)
        
        if not v3d.item():
            raise ValueError("3D Conformer generation failed")

        # 3. 关键点：从加氢后的分子中，反向找回原分子的原子索引
        # 因为 mol 是 RemoveHs 过的，mol_with_h 是 AddHs 来的
        # GetSymbol != 'H' 的索引顺序和数量必须严格等于 data.x
        heavy_idx = [a.GetIdx() for a in mol_with_h.GetAtoms() if a.GetSymbol() != 'H']
        
        num_nodes = data.x.size(0)
        if len(heavy_idx) != num_nodes:
            # 这里直接报错，触发外部的 except，从而踢掉这个 SMILES
            raise ValueError(f"Index mismatch: Graph={num_nodes}, HeavyAtoms={len(heavy_idx)}")

        # 4. 赋值
        data.pos_mean = mu_all[heavy_idx][:, heavy_idx].reshape(-1)
        data.pos_std = sigma_all[heavy_idx][:, heavy_idx].reshape(-1)
        data.valid_3d = v3d
        return data

def mol2graph(mol):
    try:
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype = np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int64)

        else:   # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

        # positions
        positions = mol.GetConformer().GetPositions()

        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = x
        graph['num_nodes'] = len(x)
        graph['position'] = positions

        return graph
    except:
        return None



def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False) -> str:
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A smiles string or an RDKit molecule.
    :param include_chirality: Whether to include chirality.
    :return:
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        try:
            scaffold = generate_scaffold(mol)
        except:
            continue
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds

def scaffold_split(conf_list,
                   balanced: bool = True,
                   seed: int = 0,): 
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
    :param seed: Seed for shuffling when doing balanced splitting.
    :return: A tuple containing the train, validation, and test split index.
    """

    # Split
    dataset_len = len(conf_list)
    train_size, val_size = 0.8 * dataset_len, 0.1 * dataset_len
    test_size = dataset_len - train_size - val_size
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(conf_list, use_indices=True)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1


    print(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                    f'train scaffolds = {train_scaffold_count:,} | '
                    f'val scaffolds = {val_scaffold_count:,} | '
                    f'test scaffolds = {test_scaffold_count:,}')

    # Map from indices to data
    # train = [data[i] for i in train]
    # val = [data[i] for i in val]
    # test = [data[i] for i in test]

    # return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    return train, val, test

def random_scaffold_split(conf_list, balanced=False, seed: int = 0):
    dataset_len = len(conf_list)
    train_cutoff = 0.8 * dataset_len
    val_cutoff = 0.9 * dataset_len # 注意：这是累加比例 (0.8 train + 0.1 val)
    
    scaffold_to_indices = scaffold_to_smiles(conf_list, use_indices=True)
    index_sets = list(scaffold_to_indices.values())
    
    random.seed(seed)
    random.shuffle(index_sets)
    
    train, val, test = [], [], []
    
    for index_set in index_sets:
        current_len = len(train) + len(val) + len(test)
        
        # 逻辑：只要总数还没到 80%，就往 train 里塞
        if len(train) < train_cutoff:
            train += index_set
        # 只要总数还没到 90%，就往 val 里塞
        elif len(val) + len(train) < val_cutoff:
            val += index_set
        # 剩下的全给 test
        else:
            test += index_set

    return train, val, test

from rdkit import Chem
from rdkit.Chem import AllChem

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