import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    FairseqDropout,
)

from fairseq import utils

from torch import Tensor
from typing import Callable

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class AtomFeature(nn.Module):
    """
    Compute atom features for each atom in the molecule.
    """

    def __init__(self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers, no_2d=False):
        super(AtomFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        self.no_2d = no_2d

        # 1 for graph token
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, mask_2d=None):
        x, in_degree, out_degree = batched_data['x'],batched_data['in_degree'], batched_data['out_degree']
        n_graph, n_node = x.size()[:2]

        # node feauture + graph token
        node_feature = self.atom_encoder(x).sum(dim=-2) # [n_graph, n_node, n_hidden]

        if not self.no_2d:
            degree_feature = self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
            if mask_2d is not None:
                degree_feature = degree_feature * mask_2d[:, None, None]
            node_feature = node_feature + degree_feature

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class MoleculeAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(self, num_heads, num_atoms, num_edges, num_spatial, num_edge_dis, hidden_dim, edge_type, multi_hop_max_dist, n_layers, no_2d=False):
        super(MoleculeAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.no_2d = no_2d

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)

        self.edge_type = edge_type
        if self.edge_type == 'multi_hop':
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, mask_2d=None):
        attn_bias, spatial_pos, x = batched_data['attn_bias'], batched_data['spatial_pos'], batched_data['x']
        edge_input, attn_edge_type = batched_data['edge_input'], batched_data['attn_edge_type']

        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        if not self.no_2d:
            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
            if mask_2d is not None:
                spatial_pos_bias = spatial_pos_bias * mask_2d[:, None, None, None]
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
                                                        :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        if not self.no_2d:

            # edge feature
            if self.edge_type == 'multi_hop':
                spatial_pos_ = spatial_pos.clone()
                spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
                # set 1 to 1, x > 1 to x - 1
                spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
                if self.multi_hop_max_dist > 0:
                    spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                    edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
                # [n_graph, n_node, n_node, max_dist, n_head]
                edge_input = self.edge_encoder(edge_input).mean(-2)
                max_dist = edge_input.size(-2)
                edge_input_flat = edge_input.permute(
                    3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
                edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads)[:max_dist, :, :])
                edge_input = edge_input_flat.reshape(
                    max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
                edge_input = (edge_input.sum(-2) /
                              (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)

            else:
                # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
                edge_input = self.edge_encoder(
                    attn_edge_type).mean(-2).permute(0, 3, 1, 2)

            if mask_2d is not None:
                edge_input = edge_input * mask_2d[:, None, None, None]
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
                                                            :, 1:, 1:] + edge_input

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias


class Molecule3DBias(nn.Module):
    """
    Compute 3D attention bias.
    IMPROVED: Fuses Static 3D Geometry (Ground Truth) with Distributional Uncertainty (Mean/Std).
    Fixed: Broadcasting mismatch (Sum over neighbor dimension for edge features).
    """

    def __init__(self, num_heads, num_edges, n_layers, embed_dim, num_kernel, no_share_rpe=False):
        super(Molecule3DBias, self).__init__()
        self.num_heads = num_heads
        self.num_edges = num_edges
        self.n_layers = n_layers
        self.no_share_rpe = no_share_rpe
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim

        # === 1. 静态流 (Static Stream) - 处理原始 pos ===
        rpe_heads = self.num_heads * self.n_layers if self.no_share_rpe else self.num_heads
        self.gbf = GaussianLayer(self.num_kernel, num_edges)
        self.gbf_proj = NonLinear(self.num_kernel, rpe_heads)

        if self.num_kernel != self.embed_dim:
            self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)
        else:
            self.edge_proj = None

        # === 2. 分布流 (Distributional Stream) - 处理 pos_mean/std ===
        self.dist_gbf = GaussianLayer(self.num_kernel, num_edges)
        self.dist_attn_proj = NonLinear(self.num_kernel, rpe_heads)
        self.dist_edge_proj = nn.Linear(self.num_kernel, self.embed_dim)

        # 不确定性门控
        self.uncertainty_gate = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward_static(self, batched_data, padding_mask):
        """计算原始 Transformer-M 的静态 3D 特征"""
        pos, x = batched_data['pos'], batched_data['x']
        node_type_edge = batched_data.get('node_type_edge', None)

        # pos: [B, N, 3] -> delta_pos: [B, N, N, 3]
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1) # [B, N, N]
        
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        # 高斯编码 [B, N, N, K]
        edge_feature = self.gbf(dist, torch.zeros_like(dist).long() if node_type_edge is None else node_type_edge.long())
        
        # --- 1. Static Attention Bias ---
        gbf_result = self.gbf_proj(edge_feature) # [B, N, N, H]
        graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous() # [B, H, N, N]
        graph_attn_bias.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # --- 2. Static Edge Features (Node Update) ---
        # Mask padding neighbors (columns) before summing
        # padding_mask: [B, N] -> [B, 1, N, 1]
        edge_feature_masked = edge_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )
        
        # 【关键修复】对邻居维度 (dim=-2) 求和: [B, N, N, K] -> [B, N, K]
        sum_edge_features = edge_feature_masked.sum(dim=-2)
        
        # Project: [B, N, K] -> [B, N, D]
        merge_edge_features = self.edge_proj(sum_edge_features) if self.edge_proj else sum_edge_features
        
        return graph_attn_bias, merge_edge_features, delta_pos

    def forward_distribution(self, batched_data, padding_mask):
        """计算分布特征带来的 Bias"""
        mu = batched_data["pos_mean"]   # [B, N, N]
        sigma = batched_data["pos_std"] # [B, N, N]
        node_type_edge = batched_data.get('node_type_edge', None)

        # 高斯编码 [B, N, N, K]
        dist_feature = self.dist_gbf(mu, torch.zeros_like(mu).long() if node_type_edge is None else node_type_edge.long())
        
        # 门控 [B, N, N, 1]
        gate = self.uncertainty_gate(sigma.unsqueeze(-1))
        
        # --- 1. Distribution Attention Bias ---
        dist_bias_raw = self.dist_attn_proj(dist_feature) # [B, N, N, H]
        dist_attn_bias = dist_bias_raw * gate
        dist_attn_bias = dist_attn_bias.permute(0, 3, 1, 2).contiguous() # [B, H, N, N]
        dist_attn_bias.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), 0.0) 

        # --- 2. Distribution Edge Features (Node Update) ---
        # Mask padding neighbors (columns)
        dist_feature_masked = dist_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )
        
        # 应用 Gate 并对邻居求和: [B, N, N, K] * [B, N, N, 1] -> Sum(-2) -> [B, N, K]
        gated_dist_feature = dist_feature_masked * gate
        sum_dist_features = gated_dist_feature.sum(dim=-2)
        
        # Project: [B, N, K] -> [B, N, D]
        dist_edge_feat = self.dist_edge_proj(sum_dist_features)
        
        return dist_attn_bias, dist_edge_feat

    def forward(self, batched_data, padding_mask):
        # 1. 基础路径 (Static)
        static_attn_bias, static_edge_feat, delta_pos = self.forward_static(batched_data, padding_mask)

        # 2. 增强路径 (Distributional)
        if "pos_mean" in batched_data and batched_data["pos_mean"] is not None:
            dist_attn_bias, dist_edge_feat = self.forward_distribution(batched_data, padding_mask)
            
            # 3. 融合 (Add)
            # Shapes:
            # attn_bias: [B, H, N, N] + [B, H, N, N] -> OK
            # edge_feat: [B, N, D] + [B, N, D] -> OK (之前报错的地方就是这里)
            final_attn_bias = static_attn_bias + dist_attn_bias
            final_edge_feat = static_edge_feat + dist_edge_feat
            
            return final_attn_bias, final_edge_feat, delta_pos
        
        else:
            return static_attn_bias, static_edge_feat, delta_pos
        
        
@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=512*3):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


class AtomTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

        self.dropout_module = FairseqDropout(
            0.1, module_name=self.__class__.__name__
        )

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor,
        delta_pos: Tensor,
    ) -> Tensor:
        query = query.contiguous().transpose(0, 1)
        bsz, n_node, _ = query.size()
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        attn_probs_float = utils.softmax(attn.view(-1, n_node, n_node) + attn_bias.contiguous().view(-1, n_node, n_node), dim=-1, onnx_trace=False)
        attn_probs = attn_probs_float.type_as(attn)
        attn_probs = self.dropout_module(attn_probs).view(bsz, self.num_heads, n_node, n_node)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force