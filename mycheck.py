import torch
import numpy as np

def check_processed_pt(pt_path):
    print(f"正在读取文件: {pt_path}")
    
    # 1. 加载数据
    # PyG 的 InMemoryDataset 存储格式通常是 (data, slices)
    data_all, slices = torch.load(pt_path)

    print("数据加载成功！")
    print(data_all)
    
    # 获取总分子数
    num_molecules = slices['x'].size(0) - 1
    print(f"检测到分子总数: {num_molecules}")
    print("-" * 30)

    # 2. 抽样检查（检查前 3 个分子）
    for i in range(min(3, num_molecules)):
        print(f"\n检查第 {i} 个分子:")
        
        # 手动切片获取第 i 个分子的数据
        node_start = slices['x'][i]
        node_end = slices['x'][i+1]
        num_nodes = node_end - node_start
        
        # 获取该分子的 pos_mean
        # 注意：如果存储时被拉平了，这里切片会很复杂
        # 如果存储时是正确的 List/None 模式，slices 里应该有 pos_mean
        if 'pos_mean' in slices:
            p_start = slices['pos_mean'][i]
            p_end = slices['pos_mean'][i+1]
            p_size = p_end - p_start
            
            actual_pos_mean = data_all.pos_mean[p_start:p_end]
            
            print(f"  节点数 (N): {num_nodes}")
            print(f"  pos_mean 展平长度: {p_size}")
            
            # --- 核心逻辑验证 ---
            # 验证 1: 矩阵大小是否等于 N^2
            expected_size = num_nodes * num_nodes
            if p_size == expected_size:
                print(f"  ✅ 维度匹配: {p_size} == {num_nodes}^2")
            else:
                print(f"  ❌ 维度冲突: 期望 {expected_size}, 实际得到 {p_size}")
            
            # 验证 2: 检查是否执行了 +1 偏移 (假设你已经在存储前加了 1)
            # 如果你在 collator 里加 1，这里可能还是原始值
            print(f"  数据区域最小值: {actual_pos_mean.min().item():.4f}")
            
        else:
            print("  ⚠️ 警告: slices 中未发现 pos_mean，请确认存储逻辑。")

    # 3. 检查全局维度 (针对你提到的 1.4M 现象)
    total_x = data_all.x.size(0)
    total_p = data_all.pos_mean.size(0)
    print("\n" + "-" * 30)
    print(f"全局统计:")
    print(f"  总节点数: {total_x}")
    print(f"  pos_mean 总长度: {total_p}")
    
    # 如果 total_p == total_x ** 2，说明整个数据集被当成一个大图了（这是错误的）
    # 如果 total_p == sum(n_i ** 2)，说明存储逻辑是正确的
    print("-" * 30)

# 使用方法：将路径替换为你生成的 pt 文件路径
# check_processed_pt('/home/duanjw/workspace/OStars/ZJ/Transformer_bias_add_moleculenet/datasets/moleculenet/esol/processed/geometric_data_processed.pt')
                      
if __name__ == "__main__":
    # 替换为你实际的 processed 文件路径
    path = "/home/duanjw/workspace/OStars/ZJ/Transformer_bias_add_moleculenet/datasets/moleculenet/esol/processed/geometric_data_processed.pt"
    # deep_scan(path)
    # check_tiny_values(path, threshold=1e-4)
    # data = torch.load(path)
    # print(data[0])
    # data_0 = data[0]
    # print("--- Dataset Item Check ---")
    # print(f"Nodes (N): {data_0.x[0].size(0)}")
    # print(f"pos_mean shape: {data_0.pos_mean}")
    # print(f"pos_std shape: {data_0.pos_std}")

    check_processed_pt(path)
