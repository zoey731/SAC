import os
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.utils import remove_self_loops
from multiprocessing import Pool, cpu_count

# ==========================================
# 1. 独立的 Worker 函数 (用于多进程并行计算)
# ==========================================
def preprocess_3d_worker(args):
    """
    输入: (smiles, target_num_nodes)
    输出: (mu, sigma, valid_mask)
    """
    smiles, target_num_nodes = args
    n_confs = 10
    
    # QM9 数据集包含氢原子，所以我们必须保留氢
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, False
        
    mol = Chem.AddHs(mol) # 必须加氢，因为 newHQM9 的图包含氢
    
    # 简单的节点数校验
    if mol.GetNumAtoms() != target_num_nodes:
        # 如果节点数对不上，说明 SMILES 和 Graph 不匹配，直接失败
        return None, None, False

    # 生成构象
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True
    
    try:
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    except:
        return None, None, False

    if not cids:
        return None, None, False

    # 批量优化
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200, numThreads=1)
    except:
        pass # 忽略优化错误，尽力而为

    # 提取坐标 [K, N, 3]
    coords_list = []
    for cid in cids:
        conf = mol.GetConformer(cid)
        coords_list.append(conf.GetPositions())
    
    if len(coords_list) == 0:
        return None, None, False
        
    coords = np.array(coords_list) # [K, N, 3]

    # 向量化计算距离矩阵 [K, N, N]
    delta = coords[:, :, np.newaxis, :] - coords[:, np.newaxis, :, :]
    dist_maps = np.sqrt(np.sum(delta**2, axis=-1))

    # 计算统计量
    mu = np.mean(dist_maps, axis=0)
    sigma = np.std(dist_maps, axis=0)

    # 转为 Tensor (注意用 float32 节省显存)
    return torch.tensor(mu, dtype=torch.float), torch.tensor(sigma, dtype=torch.float), True


def get_distributional_3d_features(mol, n_confs=10):
    """
    输入 RDKit Mol 对象，计算多构象下的距离均值和标准差。
    返回: (mu, sigma, valid_mask)
    """
    num_atoms = mol.GetNumAtoms()
    
    # 初始化失败时的返回值 (全0矩阵, False)
    zeros = torch.zeros((num_atoms, num_atoms), dtype=torch.float)
    fail_result = (zeros, zeros, torch.tensor([False], dtype=torch.bool))

    try:
        # 1. 生成构象 (Embed)
        # useSmallRingTorsions 对小环分子很重要
        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        
        # 这一步会尝试生成 n_confs 个构象
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        
        # 如果一个构象都没生成出来，直接返回失败
        if len(cids) == 0:
            return fail_result

        # 2. 力场优化 (Optimize)
        # MMFFOptimizeMoleculeConfs 是 C++ 层面优化的，比 Python 循环快
        # numThreads=0 表示使用所有可用线程，但在 process 已经是单线程循环时，设为 1 避免争抢
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200, numThreads=1)
        except:
            # 即使优化报错，只要坐标还在，我们尽量利用（或者你可以选择在这里 return fail_result）
            pass

        # 3. 提取坐标并计算统计量
        coords_list = []
        for cid in cids:
            conf = mol.GetConformer(cid)
            coords_list.append(conf.GetPositions())
        
        # 转为 numpy 进行广播计算
        coords = np.array(coords_list) # Shape: [K, N, 3]
        
        # 计算距离矩阵: (x_i - x_j)^2 + ...
        # Shape 变换: [K, N, 1, 3] - [K, 1, N, 3] -> [K, N, N, 3]
        delta = coords[:, :, np.newaxis, :] - coords[:, np.newaxis, :, :]
        dist_maps = np.sqrt(np.sum(delta**2, axis=-1)) # Shape: [K, N, N]

        # 4. 计算均值和方差
        mu = np.mean(dist_maps, axis=0)
        sigma = np.std(dist_maps, axis=0)

        # 返回 Tensor
        return (torch.tensor(mu, dtype=torch.float), 
                torch.tensor(sigma, dtype=torch.float), 
                torch.tensor([True], dtype=torch.bool))

    except Exception as e:
        # 捕获所有 RDKit 可能抛出的奇怪异常，保证主循环不崩溃
        # print(f"3D gen failed: {e}") 
        return fail_result