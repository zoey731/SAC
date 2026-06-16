import os
import os.path as osp

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)
from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict)

import numpy as np

try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem.rdchem import HybridizationType
    from rdkit.Chem.rdchem import BondType as BT
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    rdkit = None


HAR2EV = 27.2113825435
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}


class newQM9(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """  # noqa: E501

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://pytorch-geometric.com/datasets/qm9_v2.zip'

    if rdkit is not None:
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(newQM9, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.train_mean = 0.0
        self.train_std = 0.0

    def mean(self, target, idxs):
        y = torch.cat([self.get(i).y for i in idxs], dim=0)
        return y[:, target].mean().item()

    def std(self, target, idxs):
        y = torch.cat([self.get(i).y for i in idxs], dim=0)
        return y[:, target].std().item()

    def atomref(self, target):
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self):
        if rdkit is None:
            return 'qm9_v2.pt'
        else:
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def processed_file_names(self):
        return 'data_v2.pt'

    def download(self):
        if rdkit is None:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)
        else:
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))

    def process(self):
        if rdkit is None:
            print('Using a pre-processed version of the dataset. Please '
                  'install `rdkit` to alternatively process the raw data.')

            self.data, self.slices = torch.load(self.raw_paths[0])
            data_list = [self.get(i) for i in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=True,
                                   sanitize=True)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            if mol is None:
                continue

            # pos
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float32)

            # atoms
            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_features_list.append(atom_to_feature_vector(atom))
            x = np.array(atom_features_list, dtype=np.int64)

            # bonds
            num_bond_features = 3 # bond type, bond stereo, is_conjugated
            if len(mol.GetBonds()) > 0:
                edges_list = []
                edge_features_list = []
                for bond in mol.GetBonds():
                    bond_i = bond.GetBeginAtomIdx()
                    bond_j = bond.GetEndAtomIdx()

                    edge_feature = bond_to_feature_vector(bond)

                    # add edges in both directions
                    edges_list.append((bond_i, bond_j))
                    edge_features_list.append(edge_feature)
                    edges_list.append((bond_j, bond_i))

                    edge_features_list.append(edge_feature)
                edge_index = np.array(edges_list, dtype=np.int64).T
                edge_attr = np.array(edge_features_list, dtype=np.int64)
            else:
                edge_index = np.empty((2, 0), dtype=np.int64)
                edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

            y = target[i].unsqueeze(0)
            name = mol.GetProp('_Name')

            # data = Data(x=x, pos=pos, edge_index=edge_index,
            #             edge_attr=edge_attr, y=y, name=name, idx=i)

            data = Data()
            data.__num_nodes__ = int(len(x))
            data.edge_index = torch.from_numpy(edge_index).to(torch.int64)
            data.edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
            data.x = torch.from_numpy(x).to(torch.int64)
            data.y = y
            data.pos = pos
            data.name = name
            data.idx = i


            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class newHQM9(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """  # noqa: E501

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://pytorch-geometric.com/datasets/qm9_v2.zip'

    if rdkit is not None:
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(newHQM9, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.extra_pos_mean, self.extra_pos_std, self.extra_valid= torch.load(self.processed_paths[0], weights_only=False)
        self.train_mean = 0.0
        self.train_std = 0.0

    def mean(self, target, idxs):
        y = torch.cat([self.get(i).y for i in idxs], dim=0)
        return y[:, target].mean().item()

    def std(self, target, idxs):
        y = torch.cat([self.get(i).y for i in idxs], dim=0)
        return y[:, target].std().item()

    def atomref(self, target):
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    def get(self, idx):
        # 1. 调用父类方法，获取 PyG 管理的基础数据 (x, edge_index, y, pos)
        # 这会自动处理 slices 逻辑
        data = super(newHQM9, self).get(idx)
        
        # 2. 将我们单独存储的变长矩阵挂载到 data 对象上
        data.pos_mean = self.extra_pos_mean[idx]
        data.pos_std = self.extra_pos_std[idx]
        data.valid_3d_dist = self.extra_valid[idx]
        
        return data
    
    @property
    def raw_file_names(self):
        if rdkit is None:
            return 'qm9_v2.pt'
        else:
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def processed_file_names(self):
        return 'data_v2_3d.pt'

    def download(self):
        if rdkit is None:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)
        else:
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))

    def process(self):
        if rdkit is None:
            print('Using a pre-processed version of the dataset. Please '
                  'install `rdkit` to alternatively process the raw data.')

            self.data, self.slices = torch.load(self.raw_paths[0])
            data_list = [self.get(i) for i in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=True)

        data_list = []
        all_pos_mean = []
        all_pos_std = []
        all_valid_mask = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            if mol is None:
                continue

            # pos
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float32)

            # atoms
            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_features_list.append(atom_to_feature_vector(atom))
            x = np.array(atom_features_list, dtype=np.int64)

            # bonds
            num_bond_features = 3 # bond type, bond stereo, is_conjugated
            if len(mol.GetBonds()) > 0:
                edges_list = []
                edge_features_list = []
                for bond in mol.GetBonds():
                    bond_i = bond.GetBeginAtomIdx()
                    bond_j = bond.GetEndAtomIdx()

                    edge_feature = bond_to_feature_vector(bond)

                    # add edges in both directions
                    edges_list.append((bond_i, bond_j))
                    edge_features_list.append(edge_feature)
                    edges_list.append((bond_j, bond_i))

                    edge_features_list.append(edge_feature)
                edge_index = np.array(edges_list, dtype=np.int64).T
                edge_attr = np.array(edge_features_list, dtype=np.int64)
            else:
                edge_index = np.empty((2, 0), dtype=np.int64)
                edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

            y = target[i].unsqueeze(0)
            name = mol.GetProp('_Name')

            # data = Data(x=x, pos=pos, edge_index=edge_index,
            #             edge_attr=edge_attr, y=y, name=name, idx=i)

            data = Data()
            data.__num_nodes__ = int(len(x))
            data.edge_index = torch.from_numpy(edge_index).to(torch.int64)
            data.edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
            data.x = torch.from_numpy(x).to(torch.int64)
            data.y = y
            data.pos = pos
            data.name = name
            data.idx = i
            
            mu, sigma, valid_mask = get_distributional_3d_features(mol, n_confs=10)

            all_pos_mean.append(mu)
            all_pos_std.append(sigma)
            all_valid_mask.append(valid_mask)
           
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        data, slices = self.collate(data_list)

        print(f"Processing complete. Saving {len(data_list)} molecules...")
        # torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save((data, slices, all_pos_mean, all_pos_std, all_valid_mask), self.processed_paths[0])

from rdkit import Chem
from rdkit.Chem import AllChem
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