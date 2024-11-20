# Copyright (c) Tencent Inc. All rights reserved.
from typing import List
import time
import torch
import math
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from timm.models.layers import DropPath
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, Linear
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmyolo.models.layers import CSPLayerWithTwoConv


@MODELS.register_module()
class BiMultiHeadAttention(nn.Module):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        stable_softmax_2d=False,
        clamp_min_for_underflow=True,
        clamp_max_for_overflow=True,
    ):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = stable_softmax_2d
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l):
        # v: (B, HW, v_dim)
        # l: (B, L, l_dim)

        bsz, tgt_len, _ = v.size() # (B, HW, v_dim)

        query_states = self.v_proj(v) * self.scale # (B, HW, emb_dim)
        key_states = self._shape(self.l_proj(l), -1, bsz) # (B, L, emb_dim) -> (B, M, L, emb_dim//M)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz) # (B, HW, emb_dim) -> (B, M, HW, emb_dim//M)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz) # (B, L, emb_dim) -> (B, M, L, emb_dim//M)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim) # (BM, -1, emb_dim//M)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) # (BM, HW, emb_dim//M)
        key_states = key_states.view(*proj_shape) # (BM, L, emb_dim//M)
        value_v_states = value_v_states.view(*proj_shape) # (BM, HW, emb_dim//M)
        value_l_states = value_l_states.view(*proj_shape) # (BM, L, emb_dim//M)

        src_len = key_states.size(1) # L
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) # (BM, HW, emb_dim//M) * (BM, L, emb_dim//M) -> (BM, HW, L)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2) # (BM, L, HW)
        attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_l = attn_weights_l.softmax(dim=-1) # (BM, L, HW)
        attn_weights_v = attn_weights.softmax(dim=-1) # (BM, HW, L)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training) # (BM, HW, L)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training) # (BM, L, HW)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states) # (BM, HW, L) * (BM, L, emb_dim//M) -> (BM, HW, emb_dim//M)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states) # (BM, L, HW) * (BM, HW, emb_dim//M) -> (BM, L, emb_dim//M)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )


        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim) # (B, M, HW, emb_dim//M)
        attn_output_v = attn_output_v.transpose(1, 2) # (B, HW, M, emb_dim//M)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim) # (B, HW, emb_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim) # (B, M, L, emb_dim//M)
        attn_output_l = attn_output_l.transpose(1, 2) # (B, L, M, emb_dim//M)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim) # (B, L, emb_dim)

        attn_output_v = self.out_v_proj(attn_output_v) # (B, HW, v_dim)
        attn_output_l = self.out_l_proj(attn_output_l) # (B, L, l_dim)

        return attn_output_v, attn_output_l

@MODELS.register_module()
class DeepFusionBlock(nn.Module):
    def __init__(
        self,
        widen_factor,
        img_channel,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        stable_softmax_2d=False,
        clamp_min_for_underflow=True,
        clamp_max_for_overflow=True,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(DeepFusionBlock, self).__init__()

        v_dim = img_channel[-1]

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim,
            l_dim=l_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            stable_softmax_2d=stable_softmax_2d,
            clamp_min_for_underflow=clamp_min_for_underflow,
            clamp_max_for_overflow=clamp_max_for_overflow,
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, img_feats: List[Tensor], text_feats: Tensor):
        img_feats = list(img_feats)
        img_feat = img_feats[-1]
        B, v_dim, H, W = img_feat.shape

        v = img_feat.view(B, v_dim, -1).transpose(1, 2) # (B, HW, v_dim)
        l = text_feats # (B, L, l_dim)

        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)

        v = v.transpose(1,2).contiguous().view(B, v_dim, H, W)
        img_feats[-1] = v
        text_feats = l

        return img_feats, text_feats


@MODELS.register_module()
class MultiScaleDeepFusionBlock(nn.Module):
    def __init__(
        self,
        widen_factor,
        img_channel,
        l_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        stable_softmax_2d=False,
        clamp_min_for_underflow=True,
        clamp_max_for_overflow=True,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(MultiScaleDeepFusionBlock, self).__init__()

        v_dims = img_channel
        self.layer_norm_v = nn.ModuleList(
            [nn.LayerNorm(v_dim) for v_dim in v_dims]
        )
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.proj_l = nn.ModuleList(
            [nn.Linear(l_dim, v_dim) for v_dim in v_dims]
        )

        self.multi_scale_attn = nn.ModuleList([
            BiMultiHeadAttention(
                v_dim=v_dim,
                l_dim=v_dim,
                embed_dim=v_dim,
                num_heads=num_heads,
                dropout=dropout,
                stable_softmax_2d=stable_softmax_2d,
                clamp_min_for_underflow=clamp_min_for_underflow,
                clamp_max_for_overflow=clamp_max_for_overflow,
            ) for v_dim in v_dims
        ])


        self.gamma_v = [nn.Parameter(init_values * torch.ones((v_dim), device='cuda'), requires_grad=True) for v_dim in v_dims]
        self.final_proj_l = nn.Linear(sum(v_dims), l_dim)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, img_feats: List[Tensor], text_feats: Tensor):
        img_feats = list(img_feats)

        vs = []
        delta_ls = []
        for i, img_feat in enumerate(img_feats):
            B, v_dim, H, W = img_feat.shape
            v = img_feat.view(B, v_dim, -1).transpose(1, 2)  # (B, HW, v_dim)
            l = text_feats  # (B, L, l_dim)

            v = self.layer_norm_v[i](v)
            l = self.layer_norm_l(l)
            l = self.proj_l[i](l) # (B, L, v_dim)

            delta_v, delta_l = self.multi_scale_attn[i](v, l)

            delta_ls.append(delta_l)

            v = v + (self.gamma_v[i] * delta_v)
            v = v.transpose(1, 2).contiguous().view(B, v_dim, H, W)
            vs.append(v)

        delta_l = torch.cat(delta_ls, dim=-1)
        delta_l = self.final_proj_l(delta_l) # (B, L, l_dim)
        text_feats = text_feats + (self.gamma_l * delta_l)

        return vs, text_feats



@MODELS.register_module()
class IdentityFusionBLock(nn.Module):
    """
    Identity Fusion Block: No fusion!
    """
    def __init__(
            self,
            widen_factor: float,
            last_img_channel: int,
    ):
        super().__init__()

    def forward(self, img_feats, text_feats):
        return img_feats, text_feats


@MODELS.register_module()
class MaxSigmoidAttnFusionBLock(nn.Module):
    """
    Image-Text concatenation fusion block
    """
    def __init__(
            self,
            widen_factor: float,
            last_img_channel: int,
            text_dimension: int,
            num_heads: int = 1,
            with_scale: bool = False,
            use_depthwise: bool = False,
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
    ):
        super().__init__()
        img_channel = int(last_img_channel * widen_factor)

        self.img_proj = nn.Conv2d(img_channel, text_dimension, kernel_size=1, padding=0, stride=1)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.num_heads = num_heads
        self.head_channels = text_dimension // num_heads
        if with_scale:
            self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        else:
            self.scale = 1.0

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.project_conv = conv(img_channel,
                                 img_channel,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, img_feats, text_feats):
        """
        img_feats: List[P3, P4, P5]
        text_feats: (B, L, text_dim)
        """
        img_feats = list(img_feats)

        img_feat = img_feats[-1]
        B, _, H, W = img_feat.shape # (B, img_channel, H, W)

        # (B, img_channel, H, W) -> (B, text_dim, H, W) -> (B, M, text_dim//M, H, W)
        i_feat = self.img_proj(img_feat).reshape(B, self.num_heads, self.head_channels, H, W)
        t_feat = text_feats.reshape(B, -1, self.num_heads, self.head_channels) # (B, L, text_dim) -> (B, L, M, text_dim//M)

        attn_weight = torch.einsum('bmchw,bnmc->bmhwn', i_feat, t_feat)  # (B, M, text_dim//M, H, W) * (B, L, M, text_dim//M) -> (B, M, H, W, L)

        attn_weight = attn_weight.max(dim=-1)[0]  # (B, M, H, W)
        attn_weight = attn_weight / (self.head_channels ** 0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale # (B, M, H, W)

        img_feat = self.project_conv(img_feat)  # (B, img_channel, H, W) -> (B, img_channel, H, W)
        img_feat = img_feat.reshape(B, self.num_heads, -1, H, W)  # (B, M, img_channel//M, H, W)
        img_feat = img_feat * attn_weight.unsqueeze(2)  # (B, M, img_channel//M, H, W) * (B, M, 1, H, W)
        img_feat = img_feat.reshape(B, -1, H, W) # (B, M, img_channel//M, H, W) -> (B, M, H, W)

        img_feats[-1] = img_feat

        return img_feats, text_feats


# FIXME: ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FIXME: ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FIXME: ------------------------------------------------------------------------------------------------------------------------------------------------------------------


@MODELS.register_module()
class MaxSigmoidAttnBlock(BaseModule):
    """Max Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None,
                 use_einsum: bool = True) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = embed_channels // num_heads
        self.use_einsum = use_einsum

        self.embed_conv = ConvModule(
            in_channels,
            embed_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None) if embed_channels != in_channels else None
        self.guide_fc = Linear(guide_channels, embed_channels)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        if with_scale:
            self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        else:
            self.scale = 1.0

        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        B, _, H, W = x.shape

        guide = self.guide_fc(guide) # (B, L, 512) -> (B, L, 256)
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels) # (B, L, 256) -> (B, L, M, 256//M)
        embed = self.embed_conv(x) if self.embed_conv is not None else x # (B, C5, H, W) -> (B, 256, H, W)
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W) # (B, M, 256//M, H, W)

        if self.use_einsum:
            attn_weight = torch.einsum('bmchw,bnmc->bmhwn', embed, guide) # (B, M, 256//M, H, W) * (B, L, M, 256//M) -> (B, M, H, W, L)
        else:
            batch, m, channel, height, width = embed.shape
            _, n, _, _ = guide.shape
            embed = embed.permute(0, 1, 3, 4, 2)
            embed = embed.reshape(batch, m, -1, channel)
            guide = guide.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(embed, guide)
            attn_weight = attn_weight.reshape(batch, m, height, width, n)

        attn_weight = attn_weight.max(dim=-1)[0] # (B, M, H, W)
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale

        x = self.project_conv(x) # (B, C5, H, W)
        x = x.reshape(B, self.num_heads, -1, H, W) # (B, M, C5//M, H, W)
        x = x * attn_weight.unsqueeze(2) # (B, M, C5//M, H, W) * (B, M, 1, H, W)
        x = x.reshape(B, -1, H, W)
        return x


@MODELS.register_module()
class RepMatrixMaxSigmoidAttnBlock(BaseModule):
    """Max Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int,
                 guide_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None,
                 use_einsum: bool = True) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads
        self.use_einsum = use_einsum

        self.embed_conv = ConvModule(
            in_channels,
            embed_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None) if embed_channels != in_channels else None
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.guide_weight = nn.Parameter(
            torch.zeros(guide_channels, embed_channels // num_heads,
                        num_heads))
        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, x: Tensor, txt_feats: Tensor = None) -> Tensor:
        """Forward process."""
        B, _, H, W = x.shape

        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)

        batch, m, channel, height, width = embed.shape
        _, n, _, _ = self.guide_weight.shape
        # can be formulated to split conv
        embed = embed.permute(0, 1, 3, 4, 2)
        embed = embed.reshape(batch, m, -1, channel)
        attn_weight = torch.matmul(embed, self.guide_weight)
        attn_weight = attn_weight.reshape(batch, m, height, width, n)

        attn_weight = attn_weight.max(dim=-1)[0]
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid()

        x = self.project_conv(x)
        x = x.reshape(B, self.num_heads, -1, H, W)
        x = x * attn_weight.unsqueeze(2)
        x = x.reshape(B, -1, H, W)
        return x


@MODELS.register_module()
class RepConvMaxSigmoidAttnBlock(BaseModule):
    """Max Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int,
                 guide_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None,
                 use_einsum: bool = True) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads
        self.use_einsum = use_einsum

        self.embed_conv = ConvModule(
            in_channels,
            embed_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None) if embed_channels != in_channels else None
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.num_heads = num_heads
        self.split_channels = embed_channels // num_heads
        self.guide_convs = nn.ModuleList(
            nn.Conv2d(self.split_channels, guide_channels, 1, bias=False)
            for _ in range(num_heads))
        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, x: Tensor, txt_feats: Tensor = None) -> Tensor:
        """Forward process."""
        B, C, H, W = x.shape

        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = list(embed.split(self.split_channels, 1))
        # Bx(MxN)xHxW (H*c=C, H: heads)
        attn_weight = torch.cat(
            [conv(x) for conv, x in zip(self.guide_convs, embed)], dim=1)
        # BxMxNxHxW
        attn_weight = attn_weight.view(B, self.num_heads, -1, H, W)
        # attn_weight = torch.stack(
        #     [conv(x) for conv, x in zip(self.guide_convs, embed)])
        # BxMxNxHxW -> BxMxHxW
        attn_weight = attn_weight.max(dim=2)[0] / (self.head_channels**0.5)
        attn_weight = (attn_weight + self.bias.view(1, -1, 1, 1)).sigmoid()
        # .transpose(0, 1)
        # BxMx1xHxW
        attn_weight = attn_weight[:, :, None]
        x = self.project_conv(x)
        # BxHxCxHxW
        x = x.view(B, self.num_heads, -1, H, W)
        x = x * attn_weight
        x = x.view(B, -1, H, W)
        return x


@MODELS.register_module()
class MaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            guide_channels: int,
            embed_channels: int,
            num_heads: int = 1,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            with_scale: bool = False,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            use_einsum: bool = True) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.attn_block = MaxSigmoidAttnBlock(self.mid_channels,
                                              self.mid_channels,
                                              guide_channels=guide_channels,
                                              embed_channels=embed_channels,
                                              num_heads=num_heads,
                                              with_scale=with_scale,
                                              conv_cfg=conv_cfg,
                                              norm_cfg=norm_cfg,
                                              use_einsum=use_einsum)

        self.fusion_time1 = 0

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)

        start_time = time.perf_counter()
        x_main.append(self.attn_block(x_main[-1], guide))
        self.fusion_time1 = time.perf_counter() - start_time

        return self.final_conv(torch.cat(x_main, 1))


@MODELS.register_module()
class RepMaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            guide_channels: int,
            embed_channels: int,
            num_heads: int = 1,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            with_scale: bool = False,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            use_einsum: bool = True) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.attn_block = RepMatrixMaxSigmoidAttnBlock(
            self.mid_channels,
            self.mid_channels,
            embed_channels=embed_channels,
            guide_channels=guide_channels,
            num_heads=num_heads,
            with_scale=with_scale,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            use_einsum=use_einsum)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))


@MODELS.register_module()
class RepConvMaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            guide_channels: int,
            embed_channels: int,
            num_heads: int = 1,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            with_scale: bool = False,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            use_einsum: bool = True) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.attn_block = RepConvMaxSigmoidAttnBlock(
            self.mid_channels,
            self.mid_channels,
            embed_channels=embed_channels,
            guide_channels=guide_channels,
            num_heads=num_heads,
            with_scale=with_scale,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            use_einsum=use_einsum)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))


@MODELS.register_module()
class ImagePoolingAttentionModule(nn.Module):

    def __init__(self,
                 image_channels: List[int],
                 text_channels: int,
                 embed_channels: int,
                 with_scale: bool = False,
                 num_feats: int = 3,
                 num_heads: int = 8,
                 pool_size: int = 3,
                 use_einsum: bool = True):
        super().__init__()

        self.text_channels = text_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.num_feats = num_feats
        self.head_channels = embed_channels // num_heads
        self.pool_size = pool_size
        self.use_einsum = use_einsum
        if with_scale:
            self.scale = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        else:
            self.scale = 1.0
        self.projections = nn.ModuleList([
            ConvModule(in_channels, embed_channels, 1, act_cfg=None)
            for in_channels in image_channels
        ])
        self.query = nn.Sequential(nn.LayerNorm(text_channels),
                                   Linear(text_channels, embed_channels))
        self.key = nn.Sequential(nn.LayerNorm(embed_channels),
                                 Linear(embed_channels, embed_channels))
        self.value = nn.Sequential(nn.LayerNorm(embed_channels),
                                   Linear(embed_channels, embed_channels))
        self.proj = Linear(embed_channels, text_channels)

        self.image_pools = nn.ModuleList([
            nn.AdaptiveMaxPool2d((pool_size, pool_size))
            for _ in range(num_feats)
        ])

        self.fusion_time2 = 0

    def forward(self, text_features, image_features):
        start_time = time.perf_counter()

        B = image_features[0].shape[0]
        assert len(image_features) == self.num_feats
        num_patches = self.pool_size**2
        mlvl_image_features = [
            pool(proj(x)).view(B, -1, num_patches)
            for (x, proj, pool
                 ) in zip(image_features, self.projections, self.image_pools)
        ]
        mlvl_image_features = torch.cat(mlvl_image_features,
                                        dim=-1).transpose(1, 2)
        q = self.query(text_features)
        k = self.key(mlvl_image_features)
        v = self.value(mlvl_image_features)

        q = q.reshape(B, -1, self.num_heads, self.head_channels)
        k = k.reshape(B, -1, self.num_heads, self.head_channels)
        v = v.reshape(B, -1, self.num_heads, self.head_channels)
        if self.use_einsum:
            attn_weight = torch.einsum('bnmc,bkmc->bmnk', q, k)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(q, k)

        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = F.softmax(attn_weight, dim=-1)
        if self.use_einsum:
            x = torch.einsum('bmnk,bkmc->bnmc', attn_weight, v)
        else:
            v = v.permute(0, 2, 1, 3)
            x = torch.matmul(attn_weight, v)
            x = x.permute(0, 2, 1, 3)
        x = self.proj(x.reshape(B, -1, self.embed_channels))

        self.fusion_time2 = time.perf_counter() - start_time
        return x * self.scale + text_features


@MODELS.register_module()
class VanillaSigmoidBlock(BaseModule):
    """Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads

        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x = self.project_conv(x)
        # remove sigmoid
        # x = x * x.sigmoid()
        return x


@MODELS.register_module()
class EfficientCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            guide_channels: int,
            embed_channels: int,
            num_heads: int = 1,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            with_scale: bool = False,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.attn_block = VanillaSigmoidBlock(self.mid_channels,
                                              self.mid_channels,
                                              guide_channels=guide_channels,
                                              embed_channels=embed_channels,
                                              num_heads=num_heads,
                                              with_scale=with_scale,
                                              conv_cfg=conv_cfg,
                                              norm_cfg=norm_cfg)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))
