# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

# def pad_spatial_pos_unsqueeze(x, padlen):
#     x = x + 1
#     xlen = x.size(0)
#     if xlen < padlen:
#         new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
#         new_x[:xlen, :xlen] = x
#         x = new_x
#     return x.unsqueeze(0)

def pad_spatial_pos_unsqueeze(x, padlen):
    """
    适配 Transformer-M 逻辑：
    1. 将原始值 +1 (预留 0 给 padding)
    2. 填充到 [padlen, padlen]
    3. 增加 Batch 维度返回 [1, padlen, padlen]
    """
    # 偏移逻辑：确保 padding 的 0 与真实数据的 0 (如有) 区分开
    x = x + 1 
    
    xlen = x.size(0)
    if xlen < padlen:
        # 使用 new_zeros 确保新 Tensor 在同一设备 (CPU/GPU) 且类型一致
        new_x = x.new_zeros([padlen, padlen]) 
        new_x[:xlen, :xlen] = x
        x = new_x
    else:
        # 如果超过 padlen (理论上不应发生)，进行截断
        x = x[:padlen, :padlen]
        
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_pos_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(-1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y,
              ) for item in items]
    idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                        for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree, # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,
    )

def collator_3d(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    # 1. 过滤超大分子
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    
    # 2. 提取数据并执行【关键还原】
    items_extracted = []
    for item in items:
        n_node = item.x.size(0)
        
        # 从 Data 对象中提取打平的一维向量 [N*N]
        p_mean = getattr(item, 'pos_mean', None)
        p_std = getattr(item, 'pos_std', None)
        
        # --- 还原形状：[N*N] -> [N, N] ---
        # 只有在向量不为空且长度匹配时才还原
        if p_mean is not None:
            # 还原为 [N, N] 矩阵
            p_mean = p_mean.view(n_node, n_node)
            p_std = p_std.view(n_node, n_node)
        
        v_3d = getattr(item, 'valid_3d', torch.tensor([False]))

        items_extracted.append((
            item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
            item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y, item.pos,
            p_mean, 
            p_std,
            v_3d,
            getattr(item, 'mean', 0.0),
            getattr(item, 'std', 1.0)
        ))
    
    # 3. 解包字段 ( zip(*...) )
    (idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, 
     xs, edge_inputs, ys, poses, pos_means_list, pos_stds_list, 
     valid_3d_list, means, stds) = zip(*items_extracted)

    # --- 以下逻辑保持与你之前的代码一致 ---
    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat([pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])

    # 4. 处理分布偏置 Padding (使用还原后的 [N, N] 矩阵)
    if pos_means_list[0] is not None:
        pos_mean = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in pos_means_list])
        pos_std = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in pos_stds_list])
        valid_3d = torch.cat(valid_3d_list)
    else:
        pos_mean = pos_std = None
        valid_3d = torch.zeros(len(items), dtype=torch.bool)

    # 5. 生成 Node Type Edge (保持原有逻辑)
    node_type_edges = []
    for idx in range(len(items)):
        current_x = xs[idx]

        node_atom_type = current_x[:, 0] 
        n_nodes = current_x.shape[0]

        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)
        node_type_edges.append(node_atom_edge.long())
    
    node_type_edge = torch.cat(node_type_edges)

    # 6. 返回字典 (包含所有 TransformerMEncoder 需要的键)
    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,
        x=x,
        edge_input=edge_input,
        y=y,
        pos=pos,
        node_type_edge=node_type_edge,
        pos_mean=pos_mean,
        pos_std=pos_std,
        valid_3d=valid_3d,
        mean=means[0],
        std=stds[0]
    )

def collator_3d_qm9(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y, item.pos,
              item.train_mean, item.train_std, item.type,
              getattr(item, 'pos_mean', None), 
              getattr(item, 'pos_std', None)
              ) for item in items]
    idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys, poses, means, stds, types, pos_means_list, pos_stds_list = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                        for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])

    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])

    # === 修改 3: 处理新增矩阵的 Padding ===
    # pad_spatial_pos_unsqueeze 这个函数正好就是把 N*N pad 成 Max*Max
    # 我们利用它来处理我们的 pos_mean 和 pos_std
    
    # 检查列表中是否真的有数据 (防止有些 item 没有这俩属性)
    if pos_means_list[0] is not None:
        pos_mean = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in pos_means_list])
        pos_std = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in pos_stds_list])
    else:
        # 如果没数据，给个 None 或者空 Tensor
        pos_mean = None
        pos_std = None

    node_type_edges = []
    for idx in range(len(items)):

        node_atom_type = items[idx][6][:, 0]
        n_nodes = items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree, # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,

        pos=pos,
        node_type_edge=node_type_edge,

        mean=means[0],
        std=stds[0],
        type=types[0],
        
        pos_mean=pos_mean,
        pos_std=pos_std,
    )


def collator_3d_molnet(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    # 1. 过滤无效样本和超大分子 (此时 items 还是 Data 对象列表)
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    if len(items) == 0:
        return None

    # 2. 统一单次循环：完成所有属性的提取与 3D 还原
    # 我们不再中间覆盖 items 变量，而是把结果存入 processed_items
    processed_items = []
    for item in items:
        n_node = item.x.size(0)
        
        # --- 3D 统计量还原 ---
        p_mean = getattr(item, 'pos_mean', None)
        p_std = getattr(item, 'pos_std', None)
        if p_mean is not None and p_mean.dim() == 1:
            p_mean = p_mean.view(n_node, n_node)
        if p_std is not None and p_std.dim() == 1:
            p_std = p_std.view(n_node, n_node)

        # --- 收集所有字段 (注意顺序要与下面的 zip 对应) ---
        processed_items.append({
            'idx': item.idx,
            'attn_bias': item.attn_bias,
            'attn_edge_type': item.attn_edge_type,
            'spatial_pos': item.spatial_pos,
            'in_degree': item.in_degree,
            'out_degree': item.out_degree,
            'x': item.x,
            'edge_input': item.edge_input[:, :, :multi_hop_max_dist, :],
            'y': item.y,
            'pos': item.pos,
            'pos_mean': p_mean,
            'pos_std': p_std,
            'mask': getattr(item, 'mask', None),
            'y_mean': item.y_mean,
            'y_std': item.y_std
        })

    # 3. 批量解包 (使用字典列表转字典方式更安全)
    # 这一步代替了之前的 zip(*items)
    idxs = [it['idx'] for it in processed_items]
    attn_biases = [it['attn_bias'] for it in processed_items]
    attn_edge_types = [it['attn_edge_type'] for it in processed_items]
    spatial_poses = [it['spatial_pos'] for it in processed_items]
    in_degrees = [it['in_degree'] for it in processed_items]
    out_degrees = [it['out_degree'] for it in processed_items]
    xs = [it['x'] for it in processed_items]
    edge_inputs = [it['edge_input'] for it in processed_items]
    ys = [it['y'] for it in processed_items]
    poses = [it['pos'] for it in processed_items]
    pos_means_list = [it['pos_mean'] for it in processed_items]
    pos_stds_list = [it['pos_std'] for it in processed_items]
    masks = [it['mask'] for it in processed_items]
    
    # 获取全局回归统计量 (取第一个即可)
    y_mean_val = processed_items[0]['y_mean']
    y_std_val = processed_items[0]['y_std']

    # 4. 确定 Batch 最大维度
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)

    # 5. 执行所有字段的 Padding
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    attn_edge_type = torch.cat([pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees])
    edge_input = torch.cat([pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses])
    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])

    # 3D 统计量填充
    if pos_means_list[0] is not None:
        pos_mean = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in pos_means_list])
        pos_std = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in pos_stds_list])
    else:
        pos_mean, pos_std = None, None

    # Mask 处理
    target_mask = torch.cat(masks) if masks[0] is not None else None

    # 6. 生成 Node Type Edge (使用 xs 中已提取的原始特征)
    node_type_edges = []
    for i in range(len(processed_items)):
        current_x = xs[i]
        n_nodes = current_x.shape[0]
        node_atom_type = current_x[:, 0] 

        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)
        node_type_edges.append(node_atom_edge.long())
    
    node_type_edge = torch.cat(node_type_edges)
    
    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=out_degree,
        x=x,
        edge_input=edge_input,
        y=y,
        target_mask=target_mask,
        pos=pos,
        node_type_edge=node_type_edge,
        pos_mean=pos_mean,
        pos_std=pos_std,
        y_mean=torch.tensor([y_mean_val], dtype=torch.float),
        y_std=torch.tensor([y_std_val], dtype=torch.float),
    )