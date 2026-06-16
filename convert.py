import pandas as pd
import pickle
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def process_mol_with_3d(smiles):
    try:
        # 1. SMILES 转 Mol 对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        
        # 2. 必须加氢，否则 3D 构象不准且容易报错
        mol = Chem.AddHs(mol)
        
        # 3. 生成 3D 构象 (ETKDG 算法)
        # ps: 对于大规模数据集，这一步最耗时
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        embed_status = AllChem.EmbedMolecule(mol, params)
        
        if embed_status == -1: # 如果嵌入失败，尝试更宽松的参数
            embed_status = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            
        if embed_status != -1:
            # 可选：简单的力场优化，让 3D 坐标更自然
            AllChem.MMFFOptimizeMolecule(mol)
            return mol
        else:
            return None
    except:
        return None

def convert_to_pkl(csv_path, property_name, save_dir):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 假设官方 CSV 第一列是 SMILES，最后一列是标签（不同数据集可能不同，需微调）
    # 以 ESOL 为例: smiles, compound name, demo_label
    # 我们需要保留 smiles 和 标签列
    
    processed_rows = []
    print("Generating 3D Conformers (this may take a while)...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row['smiles']
        mol_obj = process_mol_with_3d(smiles)
        
        if mol_obj is not None:
            # 构造符合你 Dataset 类读取习惯的字典
            new_row = {
                'smiles': smiles,
                'conf': mol_obj
            }
            # 把除了 smiles 以外的所有列（通常是标签）都带上
            for col in df.columns:
                if col != 'smiles':
                    new_row[col] = row[col]
            processed_rows.append(new_row)

    new_df = pd.DataFrame(processed_rows)
    
    # 创建目录结构：root/property/property.pkl
    target_dir = os.path.join(save_dir, property_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    save_path = os.path.join(target_dir, f"{property_name}.pkl")
    
    with open(save_path, 'wb') as f:
        pickle.dump(new_df, f)
    
    print(f"Successfully saved to {save_path}")
    print(f"Processed {len(new_df)} molecules (failed: {len(df)-len(new_df)})")

# --- 使用示例 ---
if __name__ == "__main__":
    # 假设你下载了 esol.csv
    convert_to_pkl(
        csv_path='/home/zhaoqc/Transformer_bias_add_moleculenet/datasets/molecule_14000/sider.csv', # 官方文件名通常是这个
        property_name='sider', 
        save_dir='./datasets/moleculenet/archived_with_h'
    )