import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *
import math

class MCF(nn.Module):

    def __init__(self, in_channel, dim, embed_dim, num_heads=1, stacked=2, base_size=[384, 384], stage=1):
        super(MCF, self).__init__()

        # 输入模态数量（RGB、Depth、边缘图）
        self.num_modalities = 3
        self.base_size = base_size
        self.stage = stage

        # 初始化参数
        self.ratio = 2 ** (4 - stage)
        self.stacked = stacked
        self.embed_dim = embed_dim
        self.relu = nn.ReLU(inplace=True)
        self.channel_trans = nn.ModuleList([
            Conv2d(in_channel, embed_dim, 1, bn=False) for _ in range(self.num_modalities)  # 每个模态的特征提取
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList([
            SABlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=3., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=self.ratio,)
            for i in range(stacked)])

        # 模态门控：每个模态一个学习权重
        self.modality_gate = nn.Parameter(torch.ones(self.num_modalities, 1))  # 初始化为全 1 的权重

        # 输出层
        self.conv_out1 = Conv2d(embed_dim, dim, 1)
        self.conv_out3 = Conv2d(dim, dim, 3, relu=True)
        self.conv_out4 = Conv2d(dim, 1, 1)

    def initialize(self):
        weight_init(self)

    def forward(self, *modal_inputs):
        # 1. 模态特征提取
        modal_features = [channel_trans(x) for x, channel_trans in zip(modal_inputs, self.channel_trans)]  # RGB, Depth, 边缘图
        B, C, H, W = modal_features[0].shape  # 假设所有模态的尺寸一致

        # 2. 对每个模态进行注意力操作
        modal_attentions = []
        for feature in modal_features:
            x_att = feature.reshape(B, C, -1).transpose(1, 2)  # 展平特征
            for blk in self.blocks:
                x_att = blk(x_att, H, W)  # 每个模态的 SSCA 注意力块
            x_att = x_att.transpose(1, 2).reshape(B, C, H, W)
            modal_attentions.append(x_att)

        # 3. 融合多个模态的注意力图
        modality_weights = torch.sigmoid(self.modality_gate)  # 计算每个模态的权重
        attention_weighted = sum(w * attn for w, attn in zip(modality_weights, modal_attentions))  # 加权注意力图

        # 4. 后续卷积与输出
        x = attention_weighted
        x = self.conv_out1(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)
        
        return x, out

    def _ablation(self, x_in):
        x = self.res(x_in)
        out = self.conv_out4(x)
        return x, out


class SA(nn.Module):

    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super(SA, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.ratio = sr_ratio
        self.scale = qk_scale if qk_scale is not None else dim ** -0.5
        self.spatial_reduce = nn.Sequential(
            nn.Conv2d(dim, dim, self.ratio, self.ratio),
            nn.BatchNorm2d(dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def initialize(self):
        weight_init(self)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.spatial_reduce(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SABlock(nn.Module):

    def __init__(self, dim, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(SABlock, self).__init__()

        self.norm1 = norm_layer(dim)
        self.attn = SA(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def initialize(self):
        weight_init(self)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def initialize(self):
        weight_init(self)
        
    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class DWConv(nn.Module):
    #Deepwise Convolution
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def initialize(self):
        for n, m in self.named_children():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
