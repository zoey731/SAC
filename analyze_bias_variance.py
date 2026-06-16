# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr, spearmanr

# def analyze_fixed_pairwise_data(data_path):
#     print(f"正在加载数据: {data_path} ...")
#     data_tuple = torch.load(data_path, weights_only=False)
    
#     data = data_tuple[0]
#     slices = data_tuple[1]
#     mu_list = data_tuple[2]    # 对应你代码里的 mu [N, N]
#     sigma_list = data_tuple[3] # 对应你代码里的 sigma [N, N]

#     all_sigmas = []
#     all_errors = []

#     num_molecules = len(mu_list)
    
#     for i in range(num_molecules):
#         # 1. 提取 DFT 真实距离矩阵 (需要从 pos 计算一次)
#         start_idx = slices['pos'][i]
#         end_idx = slices['pos'][i+1]
#         pos_dft = data.pos[start_idx:end_idx].numpy()
        
#         # 计算 DFT 距离矩阵 [N, N]
#         diff = pos_dft[:, np.newaxis, :] - pos_dft[np.newaxis, :, :]
#         dist_dft = np.sqrt(np.sum(diff**2, axis=-1))

#         # 2. 提取你预存的 mu 和 sigma [N, N]
#         mu = mu_list[i].numpy() if torch.is_tensor(mu_list[i]) else mu_list[i]
#         sigma = sigma_list[i].numpy() if torch.is_tensor(sigma_list[i]) else sigma_list[i]

#         # 3. 只取上三角部分 (i < j)
#         iu = np.triu_indices(mu.shape[0], k=1)
        
#         m_vals = mu[iu]
#         s_vals = sigma[iu]
#         d_vals = dist_dft[iu]

#         # 4. 计算误差
#         errors = np.abs(m_vals - d_vals)

#         all_sigmas.extend(s_vals)
#         all_errors.extend(errors)

#         if (i + 1) % 10000 == 0:
#             print(f"进度: {i + 1} / {num_molecules}")

#     all_sigmas = np.array(all_sigmas)
#     all_errors = np.array(all_errors)

#     # 过滤无效值
#     mask = (all_sigmas > 1e-6) & (all_errors < 20) # 过滤掉极少数离群点
#     all_sigmas = all_sigmas[mask]
#     all_errors = all_errors[mask]

#     # 计算相关性
#     s_corr, _ = spearmanr(all_sigmas, all_errors)
#     print(f"\n修正后的 Spearman 相关系数: {s_corr:.4f}")
#     print(all_sigmas)
#     print(all_errors)


#     # 绘图
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.hexbin(all_sigmas, all_errors, gridsize=50, cmap='viridis', bins='log')
#     plt.xlabel('Conformational Sigma (Distance Std)')
#     plt.ylabel('Error |Mean_Dist - DFT_Dist|')
#     plt.title(f'Corrected Analysis (Corr: {s_corr:.4f})')

#     plt.subplot(1, 2, 2)
#     sns.regplot(x=all_sigmas, y=all_errors, scatter=False, color='red', label='Trend')
#     # 分箱展示
#     bins = np.linspace(0, all_sigmas.max(), 15)
#     indices = np.digitize(all_sigmas, bins)
#     bin_means = [all_errors[indices == j].mean() for j in range(1, len(bins))]
#     plt.plot(bins[:-1], bin_means, 'ko--')
#     plt.xlabel('Sigma Bins')
#     plt.ylabel('Mean Absolute Error')
#     plt.tight_layout()
#     plt.savefig('final_corrected_analysis.png')

# analyze_fixed_pairwise_data("/home/zhoujie/transformer-bias-add/dataset/processed/data_v2_3d.pt")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib.font_manager as fm

# --- 解决中文显示问题 ---
# 尝试寻找系统中可用的中文字体
def set_ch_font():
    # 常见 Linux 中文字体候选列表
    target_fonts = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'SimHei', 'Heiti TC', 'STHeiti']
    
    # 获取系统中所有已安装字体的名称
    try:
        # 正确的属性名是 fontManager (大写 M)
        available_fonts = {f.name for f in fm.fontManager.ttflist}
    except AttributeError:
        # 兼容极少数旧版本
        available_fonts = {f.name for f in fm.get_fontconfig_fonts()}

    # 匹配第一个可用的字体
    for font in target_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            print(f"成功加载中文字体: {font}")
            break
    else:
        print("未找到预设中文字体，坐标轴可能会出现方块。建议安装：sudo apt-get install fonts-wqy-microhei")
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

set_ch_font()

def analyze_fixed_pairwise_data(data_path):
    print(f"正在加载数据: {data_path} ...")
    # weights_only=False 是因为数据包含自定义几何类
    data_tuple = torch.load(data_path, weights_only=False)
    
    data = data_tuple[0]
    slices = data_tuple[1]
    mu_list = data_tuple[2]
    sigma_list = data_tuple[3]

    all_sigmas = []
    all_errors = []

    num_molecules = len(mu_list)
    
    for i in range(num_molecules):
        # 1. 提取 DFT 真实坐标并计算距离
        start_idx = slices['pos'][i]
        end_idx = slices['pos'][i+1]
        pos_dft = data.pos[start_idx:end_idx].numpy()
        
        diff = pos_dft[:, np.newaxis, :] - pos_dft[np.newaxis, :, :]
        dist_dft = np.sqrt(np.sum(diff**2, axis=-1))

        # 2. 提取 mu 和 sigma
        mu = mu_list[i].numpy() if torch.is_tensor(mu_list[i]) else mu_list[i]
        sigma = sigma_list[i].numpy() if torch.is_tensor(sigma_list[i]) else sigma_list[i]

        # 3. 取上三角
        iu = np.triu_indices(mu.shape[0], k=1)
        m_vals = mu[iu]
        s_vals = sigma[iu]
        d_vals = dist_dft[iu]

        # 4. 计算绝对误差
        errors = np.abs(m_vals - d_vals)
        all_sigmas.extend(s_vals)
        all_errors.extend(errors)

        if (i + 1) % 10000 == 0:
            print(f"进度: {i + 1} / {num_molecules}")

    all_sigmas = np.array(all_sigmas)
    all_errors = np.array(all_errors)

    # 过滤离群点
    mask = (all_sigmas > 1e-6) & (all_errors < 20) 
    all_sigmas = all_sigmas[mask]
    all_errors = all_errors[mask]

    s_corr, _ = spearmanr(all_sigmas, all_errors)

    # --- 绘图部分 ---
    plt.figure(figsize=(14, 6))

    # 左图：分布密度
    plt.subplot(1, 2, 1)
    # 去掉了 colorbar
    plt.hexbin(all_sigmas, all_errors, gridsize=50, cmap='viridis', bins='log')
    plt.xlabel('Distance Std')
    plt.ylabel('Error |Mean_Dist - DFT_Dist|')
    # plt.title(f'相关性分析 (Spearman: {s_corr:.4f})')

    # 右图：趋势线
    plt.subplot(1, 2, 2)
    sns.regplot(x=all_sigmas, y=all_errors, scatter=False, color='red', label='总体趋势线')
    
    bins = np.linspace(0, all_sigmas.max(), 15)
    indices = np.digitize(all_sigmas, bins)
    bin_means = []
    valid_bins = []
    for j in range(1, len(bins)):
        if len(all_errors[indices == j]) > 0:
            bin_means.append(all_errors[indices == j].mean())
            valid_bins.append(bins[j-1])
            
    plt.plot(valid_bins, bin_means, 'ko--', label='分箱平均误差')
    plt.xlabel('Sigma 分箱 ')
    plt.ylabel('MAE')
    # plt.title('误差随 Sigma 的变化趋势')
    plt.legend()

    plt.tight_layout()
    
    # 保存为 PDF
    output_filename = 'final_analysis_fixed.pdf'
    plt.savefig(output_filename, format='pdf', dpi=300)
    print(f"PDF 文件已保存至: {output_filename}")
    plt.show()

analyze_fixed_pairwise_data("/home/zhoujie/transformer-bias-add/dataset/processed_K=10/data_v2_3d.pt")
# analyze_fixed_pairwise_data("/home/zhoujie/transformer-bias-add/dataset/processed/data_v2_3d.pt")
