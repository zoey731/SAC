import torch
import numpy as np

def export_data_for_local():
    data_path = "/home/zhoujie/transformer-bias-add/dataset/processed_K=10/data_v2_3d.pt"
    print(f"正在加载 PyTorch 数据: {data_path} ...")
    
    # 加载原始数据
    data_tuple = torch.load(data_path, weights_only=False)
    data = data_tuple[0]
    slices = data_tuple[1]
    mu_list = data_tuple[2]
    sigma_list = data_tuple[3]

    print("正在转换为脱离 PyTorch 的纯 NumPy 格式...")
    
    # 1. 提取真实坐标 pos (展平的二维数组)
    pos_np = data.pos.cpu().numpy() if torch.is_tensor(data.pos) else np.array(data.pos)
    
    # 2. 提取分子切片索引 slices['pos']
    slices_pos = slices['pos']
    slices_np = slices_pos.cpu().numpy() if torch.is_tensor(slices_pos) else np.array(slices_pos)
    
    # 3. 提取 mu 和 sigma (大小不一的矩阵列表，使用 dtype=object 存放)
    mu_np = np.array([m.cpu().numpy() if torch.is_tensor(m) else m for m in mu_list], dtype=object)
    sigma_np = np.array([s.cpu().numpy() if torch.is_tensor(s) else s for s in sigma_list], dtype=object)

    # 将所有必要数据打包压缩为一个 .npz 文件
    output_file = 'extracted_qm9_data.npz'
    np.savez_compressed(output_file, 
                        pos=pos_np, 
                        slices=slices_np, 
                        mu=mu_np, 
                        sigma=sigma_np)
                        
    print(f"✅ 数据提取完成！已保存为: {output_file}")
    print("请将此文件下载到你本地电脑中。")

if __name__ == "__main__":
    export_data_for_local()