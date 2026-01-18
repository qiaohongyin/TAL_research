import math
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 0) CBAM (for [B,C,T,V])
# =========================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        hidden = max(8, in_channels // ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T,V]
        avg = F.adaptive_avg_pool2d(x, (1, 1))
        mx  = F.adaptive_max_pool2d(x, (1, 1))
        w   = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T,V]
        avg = torch.mean(x, dim=1, keepdim=True)              # [B,1,T,V]
        mx, _ = torch.max(x, dim=1, keepdim=True)             # [B,1,T,V]
        a = torch.cat([avg, mx], dim=1)                       # [B,2,T,V]
        w = torch.sigmoid(self.conv(a))                       # [B,1,T,V]
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


# ====================================
# 1) Temporal Self-Attention Utilities
# ====================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T,1]
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # not trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # [1,T,D]


class MultiHeadTemporalAttention_(nn.Module):
    """
    时间自注意力（标准 MHSA）：输入/输出都是 [B,T,D]，返回注意力矩阵 [B,H,T,T]
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = 1.0 / math.sqrt(self.head_dim)

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_out  = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,D] -> [B,H,T,d_h]
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        Q = self._to_heads(self.wq(x))
        K = self._to_heads(self.wk(x))
        V = self._to_heads(self.wv(x))

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,H,T,T]

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_scores = attn_scores + attn_mask.view(1, 1, T, T)
            elif attn_mask.dim() == 3:
                attn_scores = attn_scores + attn_mask.view(B, 1, T, T)
            else:
                raise ValueError("attn_mask 需为 [T,T] 或 [B,T,T]")

        attn = F.softmax(attn_scores, dim=-1)         # [B,H,T,T]
        attn = self.dropout_attn(attn)

        context = torch.matmul(attn, V)               # [B,H,T,d_h]
        context = context.transpose(1, 2).contiguous().view(B, T, D)  # [B,T,D]

        out = self.wo(context)
        out = self.dropout_out(out)
        out = self.norm(out + x)                      # 残差 + LN
        return out, attn


class TransformerFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ff(x))


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadTemporalAttention_(d_model, num_heads, dropout)
        self.ffn = TransformerFFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, attn = self.mha(x, attn_mask=attn_mask)
        x = self.ffn(x)
        return x, attn


def attn_to_alpha(attn: torch.Tensor) -> torch.Tensor:
    """
    将 MHSA 的注意力矩阵 [B,H,T,T] 压成每帧权重 [B,T]（被关注程度）
    """
    A = attn.mean(dim=1)      # [B,T,T]  平均各头
    alpha = A.mean(dim=1)     # [B,T]    再平均 query 维
    return alpha


class TemporalMHSAForSTGCN(nn.Module):
    """
    ☆ 逐关节的时间自注意力 ☆
    输入 [B,C,T,V]，把每个关节视作一条长度 T 的序列：
      - [B,V,C,T] -> [B*V,T,C] -> (PE + MHSA+FFN) -> [B*V,T,D] -> [B,C,T,V]
    返回：
      alpha_mhsa: [B,T,V]  （每帧每关节的权重，来自 MHSA 压缩）
      z:          [B,C,T,V]（注意力增强后的特征）
    """
    def __init__(self, in_channels: int, d_model: Optional[int] = None, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        d_model = d_model or in_channels
        self.proj_in  = nn.Linear(in_channels, d_model) if d_model != in_channels else nn.Identity()
        self.pos      = SinusoidalPositionalEncoding(d_model)
        self.block    = SelfAttentionBlock(d_model, num_heads=num_heads, dropout=dropout)
        self.proj_out = nn.Linear(d_model, in_channels) if d_model != in_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,C,T,V]
        B, C, T, V = x.shape
        z = x.permute(0, 3, 2, 1).reshape(B * V, T, C)  # [B*V,T,C] 每关节一条时间序列

        z = self.proj_in(z)               # [B*V,T,D]
        z = self.pos(z)
        z, attn = self.block(z)           # attn: [B*V,H,T,T]

        alpha = attn_to_alpha(attn)       # [B*V,T]
        alpha = alpha.view(B, V, T).permute(0, 2, 1).contiguous()  # [B,T,V]

        z = self.proj_out(z).reshape(B, V, T, C).permute(0, 3, 2, 1).contiguous()  # [B,C,T,V]
        return attn, z


# =============================
# 2) Your ST-GCN core modules
# =============================
class SpatialGraphConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, Ks: int):
        """Spatial Graph Convolution: 1x1 conv -> split Ks -> einsum with A -> sum."""
        super().__init__()
        self.Ks = Ks
        self.conv = nn.Conv2d(in_channels, out_channels * Ks, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C, T, V]
        A: [Ks, V, V]
        return: [N, out_channels, T, V]
        """
        x = self.conv(x)                     # [N, out_channels*Ks, T, V]
        n, kc, t, v = x.size()
        x = x.view(n, self.Ks, kc // self.Ks, t, v)      # [N, Ks, out_channels, T, V]
        x = torch.einsum('nkctv,kvw->nctw', (x, A))      # [N, out_channels, T, V]
        return x.contiguous()


class TemporalConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ks: int,   # not used in conv kernel, kept for signature compatibility
        Kt: int,
        stride: int = 1,
        dropout: float = 0.5,
    ) -> None:
        """Depthwise temporal conv (time-only), keep channels."""
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, in_channels, kernel_size=(Kt, 1),
                      stride=(stride, 1), padding=((Kt - 1) // 2, 0), groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class STConvBlock(nn.Module):
    """Spatial-Temporal Conv block with learnable edge mask M and CBAM."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        Ks: int,
        Kt: int,
        num_vertex: int,
        stride: int = 1,
        dropout: float = 0.5,
        residual: bool = True,
    ):
        super().__init__()
        self.sgc = SpatialGraphConvLayer(in_channels, out_channels, Ks)
        self.cbam = CBAM(out_channels, ratio=16, kernel_size=7)

        # Learnable edge importance
        self.M = nn.Parameter(torch.ones((Ks, num_vertex, num_vertex)), requires_grad=True)

        self.tgc = TemporalConvLayer(out_channels, out_channels, Ks, Kt, stride, dropout=dropout)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.sgc(x, A * self.M)   # [B,out,T,V]
        x = self.cbam(x)
        x = self.tgc(x) + res
        return x


# ======================
# 3) ST-GCN for Segmentation
# ======================
class STGCN4Seg(nn.Module):
    """ST-GCN for frame-wise segmentation with temporal MHSA injected after stgc4."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        Ks: int,
        Kt: int,
        A: np.ndarray,
        dropout: float = 0.5,
        attn_heads: int = 8,
    ):
        super().__init__()
        A = torch.tensor(A, dtype=torch.float32, requires_grad=False)  # [Ks,V,V]
        self.register_buffer('A', A)
        num_vertex = A.size(1)

        # BN over (V*C)
        self.bn = nn.BatchNorm1d(in_channels * num_vertex)

        # STGC blocks
        self.stgc1 = STConvBlock(in_channels, 32, Ks=Ks, Kt=Kt, num_vertex=num_vertex, residual=False, dropout=dropout)
        self.stgc2 = STConvBlock(32, 32, Ks=Ks, Kt=Kt, num_vertex=num_vertex, dropout=dropout)
        self.stgc3 = STConvBlock(32, 32, Ks=Ks, Kt=Kt, num_vertex=num_vertex, dropout=dropout)
        self.stgc4 = STConvBlock(32, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex, dropout=dropout)

        # ☆ Temporal Attention after stgc4 ☆
        self.temporal_attention = TemporalMHSAForSTGCN(in_channels=128,
                                                       d_model= 128,
                                                       num_heads=attn_heads,
                                                       dropout=0.1)

        self.stgc5 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex, dropout=dropout)
        self.stgc6 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex, dropout=dropout)
        self.stgc7 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex, dropout=dropout)
        self.stgc8 = STConvBlock(64, 128, Ks=Ks, Kt=Kt, num_vertex=num_vertex, dropout=dropout)

        self.stgc9  = STConvBlock(128, 128, Ks=Ks, Kt=Kt, num_vertex=num_vertex, dropout=dropout)
        self.stgc10 = STConvBlock(128, 128, Ks=Ks, Kt=Kt, num_vertex=num_vertex, dropout=dropout)
        self.stgc11 = STConvBlock(128, 128, Ks=Ks, Kt=Kt, num_vertex=num_vertex, dropout=dropout)
        self.stgc12 = STConvBlock(128, 128, Ks=Ks, Kt=Kt, num_vertex=num_vertex, dropout=dropout)

        # Frame-wise classifier: collapse V -> 1
        self.fc = nn.Conv2d(128, num_classes, kernel_size=(1, num_vertex))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: [B, IN_CH, T, V]
        return:
          - logits: [B, num_classes, T, 1]  (frame-wise)
          - med:    [B, 128, T, V]          (最后一个 block 的特征，便于蒸馏)
          - extras: dict, 包含 alpha_mhsa [B,T,V]
        """
        x = x[:, :, :, :15] 
        B, C, T, V = x.shape

        # ---- BatchNorm over (V*C) ----
        x_bn = x.permute(0, 3, 1, 2).contiguous().view(B, V * C, T)   # [B, V*C, T]
        x_bn = self.bn(x_bn)
        x = x_bn.view(B, V, C, T).permute(0, 2, 3, 1).contiguous()    # [B,C,T,V]

        # ---- ST-GCN trunk ----
        x = self.stgc1(x, self.A)   # [B,32,T,V]
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)   # [B,64,T,V]

        # ---- Temporal MHSA (per-joint) ----

        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)
        x = self.stgc7(x, self.A)
        x = self.stgc8(x, self.A)
        raw_attn, x = self.temporal_attention(x)  # alpha:[B,T,V], x:[B,64,T,V]
        H = raw_attn.shape[1]
        teacher_attn_map = raw_attn.view(B, V, H, T, T)
        teacher_attn_map = teacher_attn_map.mean(dim=2)
        
        x = self.stgc9(x, self.A)
        x = self.stgc10(x, self.A)
        x = self.stgc11(x, self.A)
        med = self.stgc12(x, self.A)          

        logits = self.fc(med)                 
        return logits , med,teacher_attn_map














class TemporalMHSA(nn.Module):
    def __init__(self, in_channels, num_joints, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.num_joints = num_joints
        embed_dim = in_channels * num_joints

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        x0 = x.permute(0, 2, 1, 3).reshape(N, T, C * V)  # (N,T,embed)

        # MHSA block
        h = self.norm1(x0)
        h, _ = self.attn(h, h, h)
        x1 = x0 + h  # residual

        # MLP block
        h = self.norm2(x1)
        h = self.mlp(h)
        out = x1 + h  # residual

        # reshape back
        out = out.reshape(N, T, C, V).permute(0, 2, 1, 3)
        return out


class STGCN4Seg_new(nn.Module):
    def __init__(
            self,
            in_channels: int = None,
            num_classes: int = None,
            Ks: int = None,
            Kt: int = None,
            A: np.ndarray = None,
    ):
        super().__init__()
        A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()
        num_vertex = A.size(1)

        # Batch Norm
        self.bn = nn.BatchNorm1d(in_channels * num_vertex)

        # STConvBlocks
        self.stgc1 = STConvBlock(in_channels, 32, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc2 = STConvBlock(32, 32, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc3 = STConvBlock(32, 32, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc4 = STConvBlock(32, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)

        self.stgc5 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc6 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc7 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc8 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)

        self.stgc9 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc10 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc11 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc12 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)

        # ⭐ Temporal Multi-Head Self-Attention
        self.temporal_mhsa = TemporalMHSA(64, num_vertex, num_heads=8)

        # Prediction head
        self.fc = nn.Conv2d(64, num_classes, kernel_size=(1, num_vertex))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, :, :15] 
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC blocks
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)

        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)
        x = self.stgc7(x, self.A)
        x = self.stgc8(x, self.A)

        x = self.stgc9(x, self.A)
        x = self.stgc10(x, self.A)
        x = self.stgc11(x, self.A)
        x = self.stgc12(x, self.A)

        # ⭐ Temporal Transformer Attention
        x = self.temporal_mhsa(x)

        # Prediction
        x = self.fc(x)
        return x
