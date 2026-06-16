from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .multihead_attention import MultiheadAttention
from .transformer_m_layers import AtomFeature, MoleculeAttnBias, Molecule3DBias, AtomTaskHead, GaussianLayer, NonLinear
from .transformer_m_encoder_layer import TransformerMEncoderLayer

def init_params(module):

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class TransformerMEncoder(nn.Module):

    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        encoder_normalize_before: bool = False,
        apply_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        sandwich_ln: bool = False,
        droppath_prob: float = 0.0,
        add_3d: bool = False,
        num_3d_bias_kernel: int = 128,
        no_2d: bool = False,
        mode_prob: str = "0.2,0.2,0.6",
    ) -> None:

        super().__init__()
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_init = apply_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.mode_prob = mode_prob

        try:
            mode_prob = [float(item) for item in mode_prob.split(',')]
            assert len(mode_prob) == 3
            assert sum(mode_prob) == 1.0
        except:
            mode_prob = [0.2, 0.2, 0.6]
        self.mode_prob = mode_prob

        self.atom_feature = AtomFeature(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            no_2d=no_2d,
        )

        self.molecule_attn_bias = MoleculeAttnBias(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            no_2d=no_2d,
        )

        self.add_3d = add_3d
        self.molecule_3d_bias = Molecule3DBias(
            num_heads=num_attention_heads,
            num_edges=num_edges,
            n_layers=num_encoder_layers,
            embed_dim=embedding_dim,
            num_kernel=num_3d_bias_kernel,
            no_share_rpe=False,
        ) if add_3d else None

        if add_3d:
            ##引入分布特征处理组件
            self.dist_gbf = GaussianLayer(num_3d_bias_kernel, num_edges)
            self.dist_proj = NonLinear(num_3d_bias_kernel, num_attention_heads)
            self.dist_edge_proj = nn.Linear(num_3d_bias_kernel, embedding_dim)

            #不确定性门控：根据std动态调节3D信息的权重
            self.uncertainty_gate_mlp = nn.Sequential(
                nn.Linear(1, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        ##这个模块用于处理原子级别的输出，特别是当我们引入了3D信息后，可以用它来预测一些与原子相关的任务，qm9是没有的
        self.atom_proc = AtomTaskHead(embedding_dim, num_attention_heads)

        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        droppath_probs = [
            x.item() for x in torch.linspace(0, droppath_prob, num_encoder_layers)
        ]
        self.layers.extend(
            [
                self.build_transformer_m_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    sandwich_ln=sandwich_ln,
                    droppath_prob=droppath_probs[_],
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_init:
            self.apply(init_params)

    def build_transformer_m_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        sandwich_ln,
        droppath_prob,
    ):
        return TransformerMEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            sandwich_ln=sandwich_ln,
            droppath_prob=droppath_prob,
        )

    def forward(
            self,
            batched_data,
            perturb=None,
            segment_labels: torch.Tensor = None,
            last_state_only: bool = False,
            positions: Optional[torch.Tensor] = None,
            token_embeddings: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            atom_output = None
            
            # 1. 计算 Padding Mask
            data_x = batched_data["x"]
            n_mol, n_atom = data_x.size()[:2]
            # 原子部分的掩码 (B x T)
            atom_padding_mask = (data_x[:, :, 0]).eq(0) 
            # 拼接 CLS token 的掩码 (B x 1)
            cls_mask = torch.zeros(n_mol, 1, device=atom_padding_mask.device, dtype=atom_padding_mask.dtype)
            padding_mask = torch.cat((cls_mask, atom_padding_mask), dim=1)

            # 2. 模式遮掩逻辑 (训练时随机选择 2D, 3D 或混合模式)
            mask_2d = mask_3d = None
            if self.training:
                mask_choice = np.random.choice(np.arange(3), n_mol, p=self.mode_prob)
                mask = torch.tensor([[1,1] if i==0 else [1,0] if i==1 else [0,1] for i in mask_choice]).to(data_x.device)
                mask_2d, mask_3d = mask[:, 0], mask[:, 1]

            # 3. 初始原子特征提取
            if token_embeddings is not None:
                x = token_embeddings
            else:
                x = self.atom_feature(batched_data, mask_2d=mask_2d)

            if perturb is not None:
                x[:, 1:, :] += perturb

            # 4. 提取 2D 注意力偏置
            attn_bias = self.molecule_attn_bias(batched_data, mask_2d=mask_2d)

            # 5. 【核心修改】处理 3D 分布偏置逻辑
            delta_pos = None
            if self.add_3d:
                attn_bias_3d, merged_edge_features, delta_pos = self.molecule_3d_bias(
                    batched_data, atom_padding_mask
                )

                valid_3d = batched_data.get('valid_3d', torch.ones(n_mol, device=x.device).bool())
                valid_mask = valid_3d.float()

                # --- 深度加固方案 ---
                # 1. 强力裁剪 3D 注意力偏置的输出范围
                # 注意：在 FP16 下，超过 60000 就会溢出。
                # 我们将其限制在 [-100, 100] 之间，这对 Softmax 来说已经足够大了，且绝不会溢出。
                attn_bias_3d = torch.clamp(attn_bias_3d, min=-100.0, max=100.0)

                if "pos_mean" in batched_data and batched_data["pos_mean"] is not None:
                    pos_mean = batched_data["pos_mean"]
                    pos_std = batched_data["pos_std"]
                    
                    # --- 修复 1：数值稳定性保护 ---
                    # 为 std 增加 epsilon，防止 sqrt 导数为 inf
                    safe_std = pos_std.clamp(min=1e-4).unsqueeze(-1)
                    gate = self.uncertainty_gate_mlp(safe_std)
                    gate = torch.where(torch.isnan(gate), torch.zeros_like(gate), gate)
                    
                    # --- 修复 2：确保 dist_gbf 输入合法 ---
                    dist_feat = self.dist_gbf(pos_mean.clamp(min=1e-6), batched_data['node_type_edge'])
                    dist_attn_bias = self.dist_proj(dist_feat).permute(0, 3, 1, 2) 
                    
                    # --- 修复 3：防止 gate 产生 NaN 后的传播 ---
                    # 如果 gate 意外产生了 NaN，至少不要毁掉基础的 attn_bias_3d
                    gate = torch.where(torch.isnan(gate), torch.zeros_like(gate), gate)
                    
                    actual_gate = gate.squeeze(-1).unsqueeze(1) # [B, 1, N, N]
                    attn_bias_3d = (attn_bias_3d + dist_attn_bias) * actual_gate
                    
                    dist_edge_feat = self.dist_edge_proj(dist_feat) 
                    
                    # --- 修复 4：显存与数值安全平滑 ---
                    # 使用 mean 时确保 mask 掉 Padding 部分（虽然目前是全图 mean，但要注意量级）
                    distribution_update = dist_edge_feat.mean(dim=2) * gate.squeeze(-1).mean(dim=2, keepdim=True)
                    merged_edge_features = merged_edge_features + distribution_update

                # 模式遮掩 + 数据有效性过滤
                combined_mask = valid_mask
                if mask_3d is not None:
                    combined_mask = combined_mask * mask_3d

                merged_edge_features = merged_edge_features * combined_mask[:, None, None]
                # 这种 masked_fill_ 也要注意顺序，确保不会在 NaN 上填充
                attn_bias_3d = attn_bias_3d.masked_fill(
                    ~combined_mask[:, None, None, None].bool(), 0.0
                )

                # 正式注入
                attn_bias[:, :, 1:, 1:] += attn_bias_3d
                x[:, 1:, :] += merged_edge_features * 0.01
            
            # 6. 标准 Transformer 流程
            if self.embed_scale is not None:
                x = x * self.embed_scale

            if self.quant_noise is not None:
                x = self.quant_noise(x)

            if self.emb_layer_norm is not None:
                x = self.emb_layer_norm(x)

            x = self.dropout_module(x)

            # B x T x C -> T x B x C (Fairseq 内部通常使用 Time-first)
            x = x.transpose(0, 1)

            inner_states = []
            if not last_state_only:
                inner_states.append(x)

            # 逐层计算
            for layer in self.layers:
                x, _ = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, self_attn_bias=attn_bias)
                if not last_state_only:
                    inner_states.append(x)

            if last_state_only:
                inner_states = [x]

            # 7. 返回结果
            if self.traceable:
                return torch.stack(inner_states), atom_output
            else:
                # 注意：Fairseq Task 通常期望返回 (inner_states, atom_output)
                # 这里的 atom_output 如果是性质预测任务，通常在 Task 层处理，这里保持 None 或返回 x
                return inner_states, atom_output

class TransformerMEncoderQM9(nn.Module):

    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        encoder_normalize_before: bool = False,
        apply_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        sandwich_ln: bool = False,
        droppath_prob: float = 0.0,
        add_3d: bool = False,
        num_3d_bias_kernel: int = 128,
        no_2d: bool = False,
        mode_prob: str = "0.2,0.2,0.6",
    ) -> None:

        super().__init__()
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_init = apply_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.mode_prob = mode_prob

        try:
            mode_prob = [float(item) for item in mode_prob.split(',')]
            assert len(mode_prob) == 3
            assert sum(mode_prob) == 1.0
        except:
            mode_prob = [0.2, 0.2, 0.6]
        self.mode_prob = mode_prob

        self.atom_feature = AtomFeature(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            no_2d=no_2d,
        )

        self.molecule_attn_bias = MoleculeAttnBias(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            no_2d=no_2d,
        )

        self.add_3d = add_3d
        #保留原始的3D bias模块作为兜底或预训练权重来源
        self.molecule_3d_bias = Molecule3DBias(
            num_heads=num_attention_heads,
            num_edges=num_edges,
            n_layers=num_encoder_layers,
            embed_dim=embedding_dim,
            num_kernel=num_3d_bias_kernel,
            no_share_rpe=False,
        ) if add_3d else None

        #定义新的分布偏置组件
        if add_3d:
            # 复用 GaussianLayer 来处理 Mean Distance
            # 这能保证我们使用和原模型一致的高斯核机制
            self.dist_gbf = GaussianLayer(num_3d_bias_kernel, num_edges)
            
            # 投影层: Kernel -> Attention Heads
            self.dist_proj = NonLinear(num_3d_bias_kernel, num_attention_heads)
            
            # 投影层: Kernel -> Embedding Dim (用于更新 Node Features)
            self.dist_edge_proj = nn.Linear(num_3d_bias_kernel, embedding_dim)

            # 核心创新：不确定性门控网络
            # Std (1 dim) -> Gate (1 dim)
            self.uncertainty_gate_mlp = nn.Sequential(
                nn.Linear(1, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid() 
            )

        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        droppath_probs = [
            x.item() for x in torch.linspace(0, droppath_prob, num_encoder_layers)
        ]

        self.layers.extend(
            [
                self.build_transformer_m_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    sandwich_ln=sandwich_ln,
                    droppath_prob=droppath_probs[_],
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_init:
            self.apply(init_params)

    def build_transformer_m_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        sandwich_ln,
        droppath_prob,
    ):
        return TransformerMEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            sandwich_ln=sandwich_ln,
            droppath_prob=droppath_prob,
        )

    def forward(
        self,
        batched_data,
        perturb=None,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        atom_output = None
        is_tpu = False
        # compute padding mask. This is needed for multi-head attention  

        data_x = batched_data["x"]
        n_mol, n_atom = data_x.size()[:2]
        padding_mask = (data_x[:,:,0]).eq(0) # B x T x 1
        padding_mask_cls = torch.zeros(n_mol, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        # B x (T+1) x 1
        # mask_dict = {0: [1, 1], 1: [1, 0], 2: [0, 1]}
        mask_2d = mask_3d = None
        # if self.training:
        #     mask_choice = np.random.choice(np.arange(3), n_mol, p=self.mode_prob)
        #     mask = torch.tensor([mask_dict[i] for i in mask_choice]).to(batched_data['pos'])
        #     mask_2d = mask[:, 0]
        #     mask_3d = mask[:, 1]

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.atom_feature(batched_data, mask_2d=mask_2d)

        if perturb is not None:
            x[:, 1:, :] += perturb

        # x: B x T x C

        attn_bias = self.molecule_attn_bias(batched_data, mask_2d=mask_2d)

        delta_pos = None

        # 检查是否包含我们的 Distributional 3D 数据
        has_distributional_bias = (
            "pos_mean" in batched_data and 
            "pos_std" in batched_data and 
            batched_data["pos_mean"] is not None
        )

        if self.add_3d:
            # 直接调用封装好的 molecule_3d_bias 模块
            # 我们只传递数据，不在这里写具体的计算公式
            attn_bias_3d, merged_edge_features, delta_pos = self.molecule_3d_bias(
                batched_data, padding_mask[:, 1:] # 传入原子部分的 padding_mask
            )

            # 统一的注入逻辑
            attn_bias[:, :, 1:, 1:] = attn_bias[:, :, 1:, 1:] + attn_bias_3d
            x[:, 1:, :] = x[:, 1:, :] + merged_edge_features * 0.01
            
            # 兼容原有的 force head (如果有的话)
            self.cur_delta_pos = delta_pos

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, self_attn_bias=attn_bias)
            if not last_state_only:
                inner_states.append(x)

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), atom_output
        else:
            if atom_output is None:
                atom_output = x  # 确保有值返回
            return inner_states, atom_output
