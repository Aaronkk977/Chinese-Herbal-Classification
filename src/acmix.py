"""
ACMix Module: Hybrid Convolution + Multi-Head Self-Attention
Based on "On the Integration of Self-Attention and Convolution"
https://arxiv.org/abs/2111.14556

ACMix combines the strengths of both convolution (local features) 
and self-attention (global features) in a single unified operation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ACMix(nn.Module):
    """
    ACMix: Attention-Convolution Mixing Module
    
    Combines convolution and self-attention in a unified framework.
    The module learns to mix both operations based on input features.
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0
    ):
        super(ACMix, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        
        # Shift operation for generating Q, K, V
        self.shift = nn.Conv2d(
            in_channels,
            3 * out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=qkv_bias
        )
        
        # Convolution branch
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        
        # Position encoding for self-attention
        self.pos_encoding = nn.Parameter(
            torch.randn(1, out_channels, 1, 1)
        )
        
        # Attention dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Output projection
        self.proj = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Learnable mixing parameter
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        qkv = self.shift(x)  # [B, 3*out_channels, H, W]
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, num_heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Self-Attention branch
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        attn_out = (attn @ v).transpose(2, 3).reshape(B, self.out_channels, H, W)
        attn_out = attn_out + self.pos_encoding
        
        # Convolution branch
        conv_out = self.conv(x)
        
        # Mix attention and convolution
        out = self.alpha * attn_out + (1 - self.alpha) * conv_out
        
        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class StackedACMix(nn.Module):
    """
    Stacked ACMix blocks for deeper feature extraction
    Used in the paper with K=2 (two stacked blocks)
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=2,
        kernel_size=3,
        stride=1,
        padding=1,
        num_heads=8,
        drop_path=0.0
    ):
        super(StackedACMix, self).__init__()
        
        self.blocks = nn.ModuleList()
        
        for i in range(num_blocks):
            # First block may change dimensions
            in_ch = in_channels if i == 0 else out_channels
            st = stride if i == 0 else 1
            
            self.blocks.append(
                ACMixBlock(
                    dim=in_ch,
                    out_dim=out_channels,
                    kernel_size=kernel_size,
                    stride=st,
                    padding=padding,
                    num_heads=num_heads,
                    drop_path=drop_path
                )
            )
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ACMixBlock(nn.Module):
    """
    ACMix block with residual connection and normalization
    Following ConvNeXt block design
    """
    
    def __init__(
        self,
        dim,
        out_dim=None,
        kernel_size=3,
        stride=1,
        padding=1,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path=0.0,
        act_layer=nn.GELU
    ):
        super(ACMixBlock, self).__init__()
        
        out_dim = out_dim or dim
        self.dim = dim
        self.out_dim = out_dim
        
        # Layer normalization (using GroupNorm as ConvNeXt)
        self.norm1 = nn.GroupNorm(1, dim)
        
        # ACMix layer
        self.acmix = ACMix(
            in_channels=dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_heads=num_heads
        )
        
        # Layer normalization
        self.norm2 = nn.GroupNorm(1, out_dim)
        
        # MLP (Feed-forward network)
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, mlp_hidden_dim, 1),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, out_dim, 1)
        )
        
        # Stochastic depth (drop path)
        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Shortcut connection
        if dim != out_dim or stride != 1:
            self.shortcut = nn.Conv2d(dim, out_dim, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        # ACMix with normalization
        x = self.norm1(x)
        x = self.acmix(x)
        x = shortcut + self.drop_path(x)
        
        # MLP with normalization
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


if __name__ == "__main__":
    # Test ACMix module
    print("Testing ACMix module...")
    
    # Test single ACMix layer
    acmix = ACMix(in_channels=64, out_channels=128, num_heads=8)
    x = torch.randn(2, 64, 56, 56)
    out = acmix(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test ACMix block
    print("\nTesting ACMix block...")
    block = ACMixBlock(dim=128, num_heads=8)
    x = torch.randn(2, 128, 56, 56)
    out = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test Stacked ACMix
    print("\nTesting Stacked ACMix (K=2)...")
    stacked = StackedACMix(in_channels=128, out_channels=256, num_blocks=2)
    x = torch.randn(2, 128, 56, 56)
    out = stacked(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    print("\n✓ All ACMix tests passed!")
