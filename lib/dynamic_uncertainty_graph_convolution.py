import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch_geometric.nn import GCNConv
from .modules import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DynamicUncertaintyGCN(nn.Module):
    def __init__(self, embed_dim, k_neighbors=8):
        super().__init__()
        self.k = k_neighbors
        
        # 叠三层GCN
        self.gcn1 = GCNConv(embed_dim, embed_dim)
        self.gcn2 = GCNConv(embed_dim, embed_dim)
        self.gcn3 = GCNConv(embed_dim, embed_dim)

        self.uncertainty_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )

    def build_graph(self, fea):
        B, C, H, W = fea.shape
        N = H * W
        coord = torch.stack(torch.meshgrid(
            torch.arange(H, device=fea.device),
            torch.arange(W, device=fea.device),
            indexing='ij'
        ), dim=-1).float().view(N, 2)
        
        fea_flat = fea.view(B, C, N).permute(0, 2, 1)
        spatial_dist = torch.cdist(coord, coord)
        feature_dist = torch.cdist(fea_flat.mean(0), fea_flat.mean(0))
        combined_dist = 0.7 * spatial_dist + 0.3 * feature_dist
        
        _, topk_indices = torch.topk(combined_dist, k=self.k, dim=1, largest=False)
        edge_index = torch.stack([
            torch.arange(N, device=fea.device).repeat_interleave(self.k),
            topk_indices.view(-1)
        ])
        return edge_index

    def forward(self, fea):
        B, C, H, W = fea.shape
        edge_index = self.build_graph(fea)
        
        fea_flat = fea.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        
        fea_gcn = fea_flat
        fea_gcn = fea_gcn + F.relu(self.gcn1(fea_gcn, edge_index))
        fea_gcn = fea_gcn + F.relu(self.gcn2(fea_gcn, edge_index))
        fea_gcn = fea_gcn + F.relu(self.gcn3(fea_gcn, edge_index))

        uncertainty = self.uncertainty_net(fea_gcn).view(B, 1, H, W)
        return fea * (1 + uncertainty)


class DUGC(nn.Module):
    def __init__(self, in_channel, out_channel, dim1=None, dim2=None, dim3=None,
                embed_dim=384, num_heads=6, gcn_k=8, **kwargs):
        super().__init__()
        self.dim = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 动态图卷积模块
        self.unc_gcn = DynamicUncertaintyGCN(embed_dim, k_neighbors=gcn_k)
        
        # 通道注意力与变换
        self.ca = SE(dim=self.dim)
        self.ct = Conv2d(self.dim, embed_dim, 1)
        
        # 初始化跨层交互模块
        self._init_interaction_modules(dim1, dim2, dim3, embed_dim)
        
        # 输出投影
        self.proj = nn.Sequential(
            Conv2d(embed_dim, embed_dim, 3),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            Conv2d(embed_dim, out_channel, 1)
        )

    def _init_interaction_modules(self, dim1, dim2, dim3, embed_dim):
        """带通道适配的交互模块初始化"""
        self.dim1, self.dim2, self.dim3 = dim1, dim2, dim3
        
        # 对dim1的交互处理
        if dim1:
            self.channel_adapter1 = Conv2d(dim1, embed_dim, 1)  # 新增通道适配
            self.interact1 = nn.Sequential(
                Conv2d(embed_dim*2, embed_dim, 3, 1, 1, relu=True),
                Conv2d(embed_dim, embed_dim, 3, 1, 1, relu=False),
            )
        
        # 对dim2的交互处理
        if dim2:
            self.channel_adapter2 = Conv2d(dim2, embed_dim, 1)  # 新增通道适配
            self.interact2 = nn.Sequential(
                Conv2d(embed_dim*2, embed_dim, 3, 1, 1, relu=True),
                Conv2d(embed_dim, embed_dim, 3, 1, 1, relu=False),
            )
        
        # 对dim3的CrossAttention处理
        if dim3:
            self.channel_adapter3 = nn.Linear(dim3, embed_dim)  # 新增通道适配
            self.interact3 = CrossAttention(
                dim1=embed_dim, 
                dim2=embed_dim,  # 适配后维度
                dim=embed_dim,
                num_heads=self.num_heads,  # 使用类属性
                qkv_bias=False
            )

    def forward(self, fea, fea_1=None, fea_2=None, fea_3=None):
        # 基础特征处理
        fea = self.ca(fea)
        fea = self.ct(fea)
        fea = self.unc_gcn(fea)
        
        # 处理dim1特征
        if self.dim1 and fea_1 is not None:
            fea_1 = self.channel_adapter1(fea_1)  # 通道适配
            fea_1 = F.interpolate(fea_1, fea.shape[2:], mode='bilinear', align_corners=False)
            fea_1 = self.interact1(torch.cat([fea, fea_1], dim=1))
            fea = fea + fea_1
        
        # 处理dim2特征
        if self.dim2 and fea_2 is not None:
            fea_2 = self.channel_adapter2(fea_2)  # 通道适配
            fea_2 = F.interpolate(fea_2, fea.shape[2:], mode='bilinear', align_corners=False)
            fea_2 = self.interact2(torch.cat([fea, fea_2], dim=1))
            fea = fea + fea_2
        
        # 处理dim3特征
        if self.dim3 and fea_3 is not None:
            B, C, H, W = fea_3.shape
            fea_3 = fea_3.view(B, C, -1).permute(0, 2, 1)
            fea_3 = self.channel_adapter3(fea_3)  # 通道适配 [B, N, embed_dim]
            fea_3 = self.interact3(fea.flatten(2).permute(0,2,1), fea_3)
            fea = fea + fea_3.permute(0,2,1).view_as(fea)
        
        return self.proj(fea)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




