import torch
import torch.nn as nn 
import timm


""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, checkpoint_seq
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch8_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),

    'vit_large_patch14_224': _cfg(url=''),
    'vit_huge_patch14_224': _cfg(url=''),
    'vit_giant_patch14_224': _cfg(url=''),
    'vit_gigantic_patch14_224': _cfg(url=''),


    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch8_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
    'vit_huge_patch14_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub_id='timm/vit_huge_patch14_224_in21k',
        num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_224_sam': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz'),
    'vit_base_patch16_224_sam': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz'),

    # DINO pretrained - https://arxiv.org/abs/2104.14294 (no classifier head, for fine-tune only)
    'vit_small_patch16_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_small_patch8_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch16_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch8_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),


    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear',
    ),

    'vit_base_patch16_rpn_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_base_patch16_rpn_224-sw-3b07e89d.pth'),

    # experimental (may be removed)
    'vit_base_patch32_plus_256': _cfg(url='', input_size=(3, 256, 256), crop_pct=0.95),
    'vit_base_patch16_plus_240': _cfg(url='', input_size=(3, 240, 240), crop_pct=0.95),
    'vit_small_patch16_36x1_224': _cfg(url=''),
    'vit_small_patch16_18x2_224': _cfg(url=''),
    'vit_base_patch16_18x2_224': _cfg(url=''),
}

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Attention_3_branches(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        #TODO MAKE THIS A FLAG JUST IN CASE
        # self.cross_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn = None
    def forward(self, x, x2, use_attn=True):
        B, N, C = x2.shape
        if x is None:
            qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]   # make torchscript happy (cannot use tensor as tuple)

            attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            self.attn = attn2
            attn2 = self.attn_drop(attn2)
            
            x2 = ( attn2 @ v2 ) if use_attn else v2

            x2 = x2.transpose(1, 2).reshape(B, N, C)
            x2 = self.proj(x2)
            x2 = self.proj_drop(x2)
            attn, attn3, attn4 = None, None, None
            v = None
            x, x3, x4 = None, None, None
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

            qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]   # make torchscript happy (cannot use tensor as tuple)
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
            attn3 = (q @ k2.transpose(-2, -1)) * self.scale
            attn4 = (q2 @ k.transpose(-2, -1)) * self.scale

            attn = attn.softmax(dim=-1)
            attn2 = attn2.softmax(dim=-1)
            attn3 = attn3.softmax(dim=-1)
            attn4 = attn4.softmax(dim=-1)
            self.attn = attn
            attn = self.attn_drop(attn)
            attn2 = self.attn_drop(attn2)
            attn3 = self.attn_drop(attn3)
            attn4 = self.attn_drop(attn4)

            x = ( attn @ v ) if use_attn else v
            x2 = ( attn2 @ v2 ) if use_attn else v2
            x3 = ( attn3 @ v2 ) if use_attn else v2
            x4 = ( attn4 @ v ) if use_attn else v

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            x2 = x2.transpose(1, 2).reshape(B, N, C)
            x2 = self.proj(x2)
            x2 = self.proj_drop(x2)

            x3 = x3.transpose(1, 2).reshape(B, N, C)
            # x3 = self.cross_proj(x3)
            x3 = self.proj(x3)
            x3 = self.proj_drop(x3)

            x4 = x4.transpose(1, 2).reshape(B, N, C)
            # x4 = self.cross_proj(x4)
            x4 = self.proj(x4)
            x4 = self.proj_drop(x4)

        return [x, x2, x3, x4], [attn, attn2, attn3, attn4], [v, v2]
        # return x, x2, x3, attn3
  

class Block_3_branches(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_3_branches(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x, x2, x1_x2_fusion, x2_x1_fusion, use_attn=True, domain_norm=False):
        if x is not None:
            xs, atts, vs = self.attn(self.norm1(x),self.norm1(x2))
            xa_attn, xa_attn2, xa_attn3, xa_attn4 = xs

            xa = x + self.drop_path(xa_attn)
            xb = x2 + self.drop_path(xa_attn2)
            
            xa = xa + self.drop_path(self.mlp(self.norm2(xa)))
            xb = xb + self.drop_path(self.mlp(self.norm2(xb)))

            xab = x1_x2_fusion + self.drop_path(xa_attn3)
            xab = xab + self.drop_path(self.mlp(self.norm2(xab)))
            xba = x2_x1_fusion + self.drop_path(xa_attn4)
            xba = xba + self.drop_path(self.mlp(self.norm2(xba)))
            
        else:
            xs, atts, vs = self.attn(None,self.norm1(x2))
            _, xa_attn2, _, _ = xs
            xb = x2 + self.drop_path(xa_attn2)
            xb = xb + self.drop_path(self.mlp(self.norm2(xb)))
            xa, xab, xba = None, None, None

        return [xa, xb, xab, xba], atts, vs


class ResPostBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.init_values = init_values

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x):
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class ParallelBlock(nn.Module):

    def __init__(
            self, dim, num_heads, num_parallel=2, mlp_ratio=4., qkv_bias=False, init_values=None,
            drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('attn', Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))
            self.ffns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('mlp', Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))

    def _forward_jit(self, x):
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x):
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x):
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)

class DeciderBlock(nn.Module):
    def __init__(self, dim, num_patches, num_prefix, drop=0.,norm_layer=nn.LayerNorm, use_norm=True):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.num_prefix = num_prefix
        self.norm = norm_layer(dim//8) if use_norm else nn.Identity()
        # self.conv1 = nn.Conv2d(self.dim, self.dim//2, 3, stride=2)
        # self.conv1_bn = nn.BatchNorm2d(self.dim//2)
        # self.conv2 = nn.Conv2d(self.dim//2, self.dim//4, 3, stride=2)
        # self.conv2_bn = nn.BatchNorm2d(self.dim//4)
        # self.conv3 = nn.Conv2d(self.dim//4, self.dim//8, 2)
        self.decider_emb_dim = 256
        self.conv1 = nn.Conv2d(self.dim, self.decider_emb_dim, 1, stride=1)
        self.conv1_bn = nn.BatchNorm2d(self.decider_emb_dim)
        self.conv2 = nn.Conv2d(self.decider_emb_dim, self.decider_emb_dim, 4, stride=2, padding=0)
        self.conv2_bn = nn.BatchNorm2d(self.decider_emb_dim)
        self.conv3 = nn.Conv2d(self.decider_emb_dim, self.decider_emb_dim, 4, stride=2, padding=0)
        self.conv3_bn = nn.BatchNorm2d(self.decider_emb_dim)
        # self.conv4 = nn.Conv2d(self.decider_emb_dim, self.decider_emb_dim, 2, stride=2, padding=0)

    def forward_pass(self, x):
        #k 1, s1 no pad, B, 128, 14, 14 (nn.Conv2d(self.dim, 64, 1, stride=1, padding=0))
        # (nn.Conv2d(64, 64, 4, stride=2, padding=0))
        # (nn.Conv2d(64, 64, 4, stride=2, padding=0))
        # (nn.Conv2d(64, 64, 2, stride=2, padding=0))
        #B 64, 2, 2 -> mean -> b, 256
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.relu(self.conv1_bn(x))
        x = self.conv2(x)
        x = F.relu(x)
        # x = F.relu(self.conv2_bn(x))
        x = self.conv3(x)
        # x = F.relu(x)
        # x = F.relu(self.conv3_bn(x))
        # x = self.conv4(x)
        # x = self.norm(x.flatten(1))
        # x = F.normalize(x.flatten(1))
        x = F.normalize(x.flatten(2).mean(-1))
        
        return x
        

    def forward(self, feats):
        # x = torch.cat(feats[:2], dim = -1).permute(0, 2, 1)[..., 1:]
        #CHECK THIS LOGIC
        if len(feats) == 2:
            feats_a, feats_b = [feat.permute(0, 2, 1)[..., self.num_prefix:] for feat in feats[:2]]
        #TRY TAKING THE CROSS FEATS
        else:
            feats_a, feats_b = [feat.permute(0, 2, 1)[..., self.num_prefix:] for feat in feats[2:]]

        # feats_a, feats_b = torch.stack(feats[:2], dim = 0).permute(0, 1, 3, 2)[..., self.num_prefix:]
        feats_a = feats_a.reshape(feats_a.shape[0], feats_a.shape[1], self.num_patches, self.num_patches)
        feats_b = feats_b.reshape(feats_b.shape[0], feats_b.shape[1], self.num_patches, self.num_patches)
        # decider_output = torch.empty(feats_a.shape[0]*feats_b.shape[0], self.dim//8, requires_grad=True).to(feats_a.device)
        #CROSS DOMAIN
        decider_output = []
        for i, a_feat in enumerate(feats_a):
            a_feat_batch = a_feat.unsqueeze(0).repeat(feats_b.shape[0], 1, 1, 1)
            # if torch.randn(1) > 0.5:
            #     x = torch.cat([a_feat_batch, feats_b], dim = 1) 
            # else:
            #     x = torch.cat([feats_b, a_feat_batch], dim = 1) 
            x = torch.cat([feats_b, a_feat_batch], dim = 1) #B, 798*2, 14, 14
            x = self.forward_pass(x)
            decider_output.append(x) #798*2 -> 64
        
        #SKETCH ONLY 
        sketch_decider_output = []
        for i, a_feat in enumerate(feats_a):
            a_feat_batch = a_feat.unsqueeze(0).repeat(feats_a.shape[0], 1, 1, 1)
            x = torch.cat([a_feat_batch, feats_a], dim = 1) 
            x = self.forward_pass(x)
            sketch_decider_output.append(x) #798*2 -> 64

        #PHOTO ONLY 
        # photo_decider_output = []
        # for i, b_feat in enumerate(feats_b):
        #     b_feat_batch = b_feat.unsqueeze(0).repeat(feats_a.shape[0], 1, 1, 1)
        #     x = torch.cat([b_feat_batch, feats_b], dim = 1) # This should be random
        #     x = self.forward_pass(x)
        #     photo_decider_output.append(x) #798*2 -> 198
        
        return torch.vstack(decider_output), torch.vstack(sketch_decider_output), None   
            

class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block_3_branches, local_feature=False, block_pattern='3_branches', use_attn=True):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
            local_features.... TODO
            block_pattern... TODO
            use_attn... TODO
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False
        
        #Added
        self.local_feature = local_feature
        self.block_pattern = block_pattern
        self.use_attn = use_attn

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        #OLD
        # self.blocks = nn.Sequential(*[
        #     block_fn(
        #         dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)])

        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'pos_embed', 'cls_token', 'dist_token'}

    # @torch.jit.ignore
    # def group_matcher(self, coarse=False):
    #     return dict(
    #         stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
    #         blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
    #     )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x, x2, domain_norm=False):
        B = x2.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        if x is not None:
            x = self.patch_embed(x)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

        x2 = self.patch_embed(x2)
        x2 = torch.cat((cls_tokens, x2), dim=1)
        x2 = x2 + self.pos_embed
        x2 = self.pos_drop(x2)
        
        if self.block_pattern == '3_branches':
            x1_x2_fusion = x2
            if x is None:
                #WILL NEVER BE USED BUT IS SET FOR TEST TIME
                x2_x1_fusion = x2
            else:
                x2_x1_fusion = x
            # cross_attn_list = []
            atts_list = []
            values_list = []
            for _, blk in enumerate(self.blocks):
                xs, atts, vs = blk(x, x2, x1_x2_fusion, x2_x1_fusion, use_attn=self.use_attn, domain_norm=domain_norm)
                atts_list.append(atts)
                values_list.append(vs)
                x, x2, x1_x2_fusion, x2_x1_fusion = xs
            if x is None:
                x2 = self.norm(x2)
                return [None, x2, None, None], atts_list, values_list
            else:
                x = self.norm(x)
                x2 = self.norm(x2)
                x1_x2_fusion = self.norm(x1_x2_fusion)
                x2_x1_fusion = self.norm(x2_x1_fusion)
                return [x, x2, x1_x2_fusion, x2_x1_fusion], atts_list, values_list
        else:
            for i, blk in enumerate(self.blocks):
                xs, atts, vs = blk(x,x2,use_attn=self.use_attn, domain_norm=domain_norm)
                x,x2 = xs
            x = self.norm(x)
            x2 = self.norm(x2)

            return x, x2 

    def forward(self, x, x2, domain_norm=False):
        #atts with shape, [depth, batchsize, num_types_att, emb_space, im_patches**2 + 1, im_patches**2 + 1]
        #vs with shape, [num vs, batchsize, emb_space, im_patches**2 + 1, 64]
        feats, atts, vs = self.forward_features(x, x2, domain_norm)
        if isinstance(feats, list):
            outputs = []
            # for feat in feats[:-1]: #cross att list adds nothing currently
            for feat in feats: #cross att list adds nothing currently
                out = self.forward_head(feat)
                outputs.append(out)
        else:
            outputs, kd_ops = self.forward_head(feats)

        return outputs, feats

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)

        return x if pre_logits else self.head(x)


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            model.pos_embed,
            getattr(model, 'num_prefix_tokens', 1),
            model.patch_embed.grid_size
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    # _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    # ntok_new = posemb_new.shape[1]
    # if num_prefix_tokens > 0:
    #     num_prefix_tokens= num_prefix_tokens-1
    #     posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
    #     ntok_new -= num_prefix_tokens
    # else:
    #     posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    # gs_old = int(math.sqrt(len(posemb_grid)))
    # if not len(gs_new):  # backwards compatibility
    #     gs_new = [int(math.sqrt(ntok_new))] * 2
    # assert len(gs_new) >= 2
    # _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    # posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    # posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    # posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    # posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    # import pdb;pdb.set_trace()
    # return posemb
    #WE WANT TO POSITION THE NULL TOKEN AT INDEX 1
    return torch.cat([posemb[:, 0, :].unsqueeze(1), posemb[:, 1, :].unsqueeze(1), posemb[:, 1:, :]], dim=1)


def checkpoint_filter_fn(state_dict, model, adapt_layer_scale=False):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                model.pos_embed,
                0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
                model.patch_embed.grid_size
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    if not isinstance(pretrained_cfg, dict):
        pretrained_cfg = pretrained_cfg.__dict__
    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model


@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_384(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16) @ 384x384.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32)
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32) at 384x384.
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch8_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch8_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch14_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/14)
    """
    model_kwargs = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_huge_patch14_224(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_giant_patch14_224(pretrained=False, **kwargs):
    """ ViT-Giant model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_kwargs = dict(patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_giant_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_gigantic_patch14_224(pretrained=False, **kwargs):
    """ ViT-Gigantic model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_kwargs = dict(patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_gigantic_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch8_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch8_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_sam(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_sam', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224_sam(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224_sam', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224_dino(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224_dino', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch8_224_dino(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/8) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=8, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch8_224_dino', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_dino(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) /w DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_dino', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch8_224_dino(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/8) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch8_224_dino', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil_in21k(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil', pretrained=pretrained, **model_kwargs)
    return model


# Experimental models below

@register_model
def vit_base_patch32_plus_256(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32+)
    """
    model_kwargs = dict(patch_size=32, embed_dim=896, depth=12, num_heads=14, init_values=1e-5, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_plus_256', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_plus_240(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16+)
    """
    model_kwargs = dict(patch_size=16, embed_dim=896, depth=12, num_heads=14, init_values=1e-5, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_plus_240', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_rpn_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ residual post-norm
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, init_values=1e-5, class_token=False,
        block_fn=ResPostBlock, global_pool=kwargs.pop('global_pool', 'avg'), **kwargs)
    model = _create_vision_transformer('vit_base_patch16_rpn_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_36x1_224(pretrained=False, **kwargs):
    """ ViT-Base w/ LayerScale + 36 x 1 (36 block serial) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=36, num_heads=6, init_values=1e-5, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_36x1_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_18x2_224(pretrained=False, **kwargs):
    """ ViT-Small w/ LayerScale + 18 x 2 (36 block parallel) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=18, num_heads=6, init_values=1e-5, block_fn=ParallelBlock, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_18x2_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_18x2_224(pretrained=False, **kwargs):
    """ ViT-Base w/ LayerScale + 18 x 2 (36 block parallel) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=18, num_heads=12, init_values=1e-5, block_fn=ParallelBlock, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_18x2_224', pretrained=pretrained, **model_kwargs)
    return model


if __name__ == "__main__":
    # vitModel = ViTModel()
    vitModel = vit_base_patch16_224(pretrained=True)
    import pdb;pdb.set_trace()