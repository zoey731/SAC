import torch

# 替换为你真实的 .pt 文件路径（注意大小写，根据你刚才的报错日志，应该是 CHEMBL204_Ki）
pt_path = '/home/zhaoqc/Transformer_bias_add_moleculeace/datasets/moleculeace/CHEMBL214_Ki/processed/sac_geometric_data.pt'

# 1. 直接加载底层的 pt 文件
# 回忆一下：我们存的时候是 torch.save(self.collate(data_list), path)
# self.collate 返回的是 (data, slices) 两个对象
giant_data, slices = torch.load(pt_path)

print("====== 1. 巨型图 (Giant Graph) 的真实面貌 ======")
# 你会看到所有的属性都被拼接成了极长的一维或二维张量
print(giant_data)

print("\n====== 2. 核心特征的合并维度 ======")
print(f"总节点特征 (x): {giant_data.x.shape}")
print(f"总边索引 (edge_index): {giant_data.edge_index.shape}")
print(f"总目标值 (y): {giant_data.y.shape}")
print(f"悬崖标签总数 (is_cliff): {giant_data.is_cliff.shape}")

print("\n====== 3. 重点检查你的 SAC 3D 统计量 ======")
# ⚠️ 这里应该是极长的一维张量，因为我们之前做了 .reshape(-1)
print(f"展平后的总 pos_mean 维度: {giant_data.pos_mean.shape}")
print(f"展平后的总 pos_std  维度: {giant_data.pos_std.shape}")

print("\n====== 4. 切片字典 (Slices) 体检 ======")
# slices 是一个字典，记录了如何把 giant_data 切分回 2754 个分子的“说明书”
print("切片包含的键值:", slices.keys())
# 通过切片的长度，可以知道到底保存了多少个分子 (通常是 数量 + 1)
print(f"成功保存的分子总数: {len(slices['x']) - 1}")