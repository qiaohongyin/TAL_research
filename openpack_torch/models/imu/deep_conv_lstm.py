import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ---------- Positional Encoding ----------
class SinusoidalPositionalEncoding(nn.Module):
    pe: torch.Tensor
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,T,D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)



# ---------- Multi-Head Temporal Self-Attention ----------
class MultiHeadTemporalAttention_(nn.Module):

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = 1.0 / math.sqrt(self.head_dim)

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_out  = nn.Dropout(dropout)

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
            # [T,T] or [B,T,T]，屏蔽位置填 -inf
            if attn_mask.dim() == 2:
                attn_scores = attn_scores + attn_mask.view(1, 1, T, T)
            elif attn_mask.dim() == 3:
                attn_scores = attn_scores + attn_mask.view(B, 1, T, T)
            else:
                raise ValueError("attn_mask 需为 [T,T] 或 [B,T,T]")

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout_attn(attn)

        context = torch.matmul(attn, V)                           # [B,H,T,d_h]
        context = context.transpose(1, 2).contiguous().view(B, T, D)  # [B,T,D]

        out = self.wo(context)
        out = self.dropout_out(out)
        out = self.norm(out + x)  # 残差 + LN

        return out, attn


# ---------- Transformer FFN ----------
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
        return self.norm(x + self.ff(x))  # 残差 + LN


# ---------- SelfAttentionBlock = MHSA + FFN ----------
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



class DeepConvLSTM_PT(nn.Module):

    def __init__(self, in_ch=6, num_classes=11, hidden_units=128):
        super().__init__()
        blocks = []
        for i in range(4):
            in_ch_ = in_ch if i == 0 else 64
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch_, 64, kernel_size=(5, 1), padding=(2, 0)),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
            )
        self.conv2to5 = nn.ModuleList(blocks)
        self.lstm6 = nn.LSTM(64, hidden_units, batch_first=True)
        self.dropout6 = nn.Dropout(0.5)

        self.pos_enc = SinusoidalPositionalEncoding(hidden_units)
        self.sa_block = SelfAttentionBlock(hidden_units, 4)

        self.lstm7 = nn.LSTM(hidden_units, hidden_units, batch_first=True)
        self.dropout7 = nn.Dropout(0.5)
        
        self.out8 = nn.Conv2d(hidden_units, num_classes, 1)

        self.reg_lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dist_reg_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        B, CH, T, _ = x.shape
        for blk in self.conv2to5:
            x = blk(x)
        
        x = x.squeeze(3).transpose(1, 2)
        r_out, _ = self.reg_lstm(x)  
        dist_pred = self.dist_reg_head(r_out) 
        dist_pred = dist_pred.squeeze(-1)

        x1, _ = self.lstm6(x)
        x1 = self.dropout6(x1)

        z = self.pos_enc(x1)
        z, attn = self.sa_block(z)
        
        x2, _ = self.lstm7(z)
        x2 = self.dropout7(x2)
        lstm_out = x2 

        class_logits = self.out8(x2.transpose(1, 2).unsqueeze(3))
        
        return class_logits, dist_pred, attn, lstm_out


class DeepConvLSTM_PT1(nn.Module):

    def __init__(self, in_ch=6, num_classes=11, hidden_units=128):
        super().__init__()
        blocks = []
        for i in range(4):
            in_ch_ = in_ch if i == 0 else 64
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch_, 64, kernel_size=(5, 1), padding=(2, 0)),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
            )
        self.conv2to5 = nn.ModuleList(blocks)
        self.lstm6 = nn.LSTM(64, hidden_units, batch_first=True)
        self.dropout6 = nn.Dropout(0.5)

        self.pos_enc = SinusoidalPositionalEncoding(hidden_units)
        self.sa_block = SelfAttentionBlock(hidden_units, 4)

        self.lstm7 = nn.LSTM(hidden_units, hidden_units, batch_first=True)
        self.dropout7 = nn.Dropout(0.5)
        
        self.out8 = nn.Conv2d(hidden_units, num_classes, 1)

        self.reg_lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dist_reg_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        B, CH, T, _ = x.shape
        for blk in self.conv2to5:
            x = blk(x)
        
        x = x.squeeze(3).transpose(1, 2)
        r_out, _ = self.reg_lstm(x)  
        dist_pred = self.dist_reg_head(r_out) 
        dist_pred = dist_pred.squeeze(-1)

        x1, _ = self.lstm6(x)
        x1 = self.dropout6(x1)

        z = self.pos_enc(x1)
        z, attn = self.sa_block(z)
        
        x2, _ = self.lstm7(z)
        x2 = self.dropout7(x2)
        lstm_out = x2 

        class_logits = self.out8(x2.transpose(1, 2).unsqueeze(3))
        
        return class_logits, dist_pred, attn, lstm_out