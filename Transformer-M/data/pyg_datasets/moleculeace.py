import os
import os.path as osp
import torch
import pandas as pd
import numpy as np
import random
import pickle
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import AllChem
from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector)

# ===== 引入 RDKit 的标准化模块 =====
from rdkit.Chem.MolStandardize import rdMolStandardize

# --- 1. SAC 核心 3D 统计量生成器 ---
def get_distributional_3d_features(mol, n_confs=10):
    """
    计算多构象下的距离矩阵均值和标准差。
    输入需为带氢的 RDKit Mol。
    """
    num_atoms = mol.GetNumAtoms()
    zeros = torch.zeros((num_atoms, num_atoms), dtype=torch.float)
    fail_result = (zeros, zeros, torch.tensor([False], dtype=torch.bool))

    try:
        # ETKDGv3 构象生成
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        
        if len(cids) < 2: return fail_result

        # MMFF 快速优化（可选，增加几何准确性）
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=50, numThreads=1)
        except: pass

        # 提取坐标并计算距离矩阵系综 [K, N, N]
        coords = np.array([mol.GetConformer(cid).GetPositions() for cid in cids])
        # 利用广播机制计算欧氏距离
        dist_maps = np.linalg.norm(coords[:, :, np.newaxis, :] - coords[:, np.newaxis, :, :], axis=-1)

        mu = np.mean(dist_maps, axis=0)
        sigma = np.std(dist_maps, axis=0)

        return (torch.from_numpy(mu).float(), 
                torch.from_numpy(sigma).float(), 
                torch.tensor([True], dtype=torch.bool))
    except:
        return fail_result

# --- 2. 2D 图结构转换器 ---
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
        # positions = mol.GetConformer().GetPositions()

        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = x
        graph['num_nodes'] = len(x)
        # graph['position'] = positions

        return graph
    except:
        return None

# --- 3. 封装好的 MoleculeACE 数据集类 ---
class MoleculeACEDataset(InMemoryDataset):
    def __init__(self, root, property_name, n_confs=10, transform=None, pre_transform=None):
        self.property = property_name
        self.n_confs = n_confs
        super().__init__(osp.join(root, property_name), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        # ==========================================
        # 【新增】：从 split 文件中加载统计量
        # ==========================================
        split_dict = torch.load(osp.join(self.root, "split_82_cluster.pt"))
        self._y_mean = split_dict.get('y_mean', 0.0)
        self._y_std = split_dict.get('y_std', 1.0)

        # 统计标签用于 Normalization
        # y_all = self.data.y
        # self._y_mean = float(torch.mean(y_all[~torch.isnan(y_all)]))
        # self._y_std = float(torch.std(y_all[~torch.isnan(y_all)]))

    @property
    def raw_file_names(self):
        return [f"{self.property}.csv"]

    @property
    def processed_file_names(self):
        return 'sac_geometric_data.pt'

    def download(self):
        pass # 假设 CSV 已手动放入 raw 文件夹

    def process(self):
        raw_path = osp.join(self.raw_dir, self.raw_file_names[0])
        df = pd.read_csv(raw_path)
        data_list = []
        train_indices, test_indices = [], []
        #===== 实例化最大片段选择器 =====
        fragment_chooser = rdMolStandardize.LargestFragmentChooser()

        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {self.property}"):
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is None: continue
            
            # ==========================================
            # 【源头修复】：脱盐与去溶剂，只保留最大的分子片段！
            # 无论输入有几个游离的离子，这行代码跑完后，mol 只剩下主药分子。
            mol = fragment_chooser.choose(mol)
            # ==========================================

            # 1. 构建 2D 图 (重原子)
            mol_2d = Chem.RemoveHs(mol)
            graph = mol2graph(mol_2d)
            if graph is None: continue

            data = Data()
            data.x = torch.from_numpy(graph['node_feat']).long()
            data.edge_index = torch.from_numpy(graph['edge_index']).long()
            data.edge_attr = torch.from_numpy(graph['edge_feat']).long()
            data.y = torch.tensor([[row['y']]], dtype=torch.float32)
            data.is_cliff = torch.tensor([row['cliff_mol']], dtype=torch.long)
            data.smiles = row['smiles']
            # 2. SAC 增强 (加氢算 3D，映射回重原子)
            try:
                mol_3d = Chem.AddHs(mol_2d)
                mu_all, sigma_all, v3d = get_distributional_3d_features(mol_3d, self.n_confs)
                
                if not v3d.item(): raise ValueError("3D Failed")

                # # 对齐重原子索引
                # heavy_idx = [a.GetIdx() for a in mol_3d.GetAtoms() if a.GetSymbol() != 'H']

                # ==========================================
                # 【氢原子 Bug 修复】
                # RDKit 的 AddHs 永远把新加的普通氢排在原子列表最后面。
                # 所以前 n_2d_nodes 个原子，绝对就是 2D 图里原封不动的那些节点！
                # ==========================================
                n_2d_nodes = mol_2d.GetNumAtoms()
                heavy_idx = list(range(n_2d_nodes))

                # 存储为 (N, N) 矩阵，方便 Transformer-M 注意力机制读取
                data.pos_mean = mu_all[heavy_idx][:, heavy_idx].reshape(-1)
                data.pos_std = sigma_all[heavy_idx][:, heavy_idx].reshape(-1)
                data.valid_3d = v3d
            except:
                # 兜底逻辑：如果 3D 失败，填充零矩阵
                print("3D失败，填充零矩阵")
                n = data.x.size(0)
                data.pos_mean = torch.zeros((n, n)).reshape(-1)
                data.pos_std = torch.zeros((n, n)).reshape(-1)
                data.valid_3d = torch.tensor([False])

            # 3. 记录 8:2 划分
            curr_idx = len(data_list)
            if row['split'] == 'train':
                train_indices.append(curr_idx)
            else:
                test_indices.append(curr_idx)
            data_list.append(data)

        # 保存数据
        print("*********************")
        print(self.processed_paths[0])
        torch.save(self.collate(data_list), self.processed_paths[0])
        
        # 验证集切分 (从 80% train 中切出 10%)
        random.seed(42)
        random.shuffle(train_indices)
        v_idx = int(len(train_indices) * 0.1)
        
        # ==========================================
        # 【新增】：计算训练集的 y_mean 和 y_std
        # ==========================================
        train_y_vals = [data_list[i].y.item() for i in train_indices]
        y_mean = np.mean(train_y_vals)
        y_std = np.std(train_y_vals)
        print(f"\n[Data Process] 计算得到训练集统计量 -> Mean: {y_mean:.4f}, Std: {y_std:.4f}")
        
        split = {
            'train': torch.tensor(train_indices[v_idx:]),
            'valid': torch.tensor(train_indices[:v_idx]),
            'test': torch.tensor(test_indices),
            'y_mean': y_mean,  # 保存到字典中
            'y_std': y_std     # 保存到字典中
        }
        torch.save(split, osp.join(self.root, "split_82_cluster.pt"))

    def get_idx_split(self):
        return torch.load(osp.join(self.root, "split_82_cluster.pt"))

    def get(self, idx):
        data = super().get(idx)
        data.y_mean = self._y_mean
        data.y_std = self._y_std
        # ==========================================
        # 【核心修复】：将目标值 y 归一化！
        # ==========================================
        data.y = (data.y - self._y_mean) / self._y_std

        # ==========================================
        # 瞒天过海：为 Transformer-M 原生代码补上缺失的 pos
        # 维度必须是 [N, 3]，全填 0 即可
        # ==========================================
        num_nodes = data.x.size(0)
        data.pos = torch.zeros((num_nodes, 3), dtype=torch.float32)
        return data