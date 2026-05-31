# layers/network.py
import torch
from torch import nn
from typing import List
from mamba_ssm import Mamba
import torch.nn.functional as F
import math
from layers.SelfAttention_Family import ResAttention



def band_mask(L: int, radius: int, device=None) -> torch.Tensor:
    i = torch.arange(L, device=device).unsqueeze(1)
    j = torch.arange(L, device=device).unsqueeze(0)
    dist = (i - j).abs()
    allowed = dist <= radius
    return ~allowed  # True=masked

import torch
from torch import nn

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.0, use_fallback: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        from mamba_ssm import Mamba
        self.mamba = Mamba(d_model=d_model, d_state=d_state)
        self.drop = nn.Dropout(dropout)

        self.use_fallback = use_fallback
        if use_fallback:
            self.fallback_rnn = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
            self.fallback_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        x_in = x_norm.contiguous()
        orig_dtype = x_in.dtype
        if x_in.dtype != torch.float32:
            x_in = x_in.float()

        try:
            with torch.cuda.amp.autocast(enabled=False):
                y = self.mamba(x_in)
        except (TypeError, RuntimeError) as e:
            if not self.use_fallback:
                raise
            y, _ = self.fallback_rnn(x_in)
            y = self.fallback_proj(y)

        # 恢复原 dtype
        if y.dtype != orig_dtype:
            y = y.to(orig_dtype)
        return x + self.drop(y)


class TopKMoEMultiWindowAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        windows: List[int] = (4, 8, 12),
        topk: int = 2,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert topk >= 1 and topk <= len(windows), "topk 应在 [1, num_experts] 范围内"
        self.d_model = d_model
        self.n_heads = n_heads
        self.windows = list(windows)
        self.topk = topk
        self.attn_experts = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
            for _ in self.windows
        ])

        self.gate = nn.Linear(d_model, len(self.windows))

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, L, D = x.shape
        device = x.device

        logits = self.gate(x)  # [B, L, E]
        
        topk_scores, topk_idx = torch.topk(logits, k=self.topk, dim=-1)  # [B, L, K]
        mask_full = torch.full_like(logits, float('-inf'))
        mask_full.scatter_(-1, topk_idx, topk_scores)
        weights = torch.softmax(mask_full, dim=-1)  # [B, L, E]

        expert_outputs = []
        for e, (w, attn) in enumerate(zip(self.windows, self.attn_experts)):
            attn_mask = band_mask(L, radius=w, device=device)  # [L, L] 
            y, _ = attn(x, x, x, attn_mask=attn_mask)
            expert_outputs.append(y)  # [B, L, D]

        # [E, B, L, D]
        Y = torch.stack(expert_outputs, dim=0)

        W = weights.permute(2, 0, 1).unsqueeze(-1)  # [E, B, L, 1]

        out = (W * Y).sum(dim=0)  # [B, L, D]
        out = self.proj(out)
        return out

class SeasonalityExtractor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        padding_patch: str = "none",
        d_model: int = 128,
        n_heads: int = 4,
        moe_windows: List[int] = (4, 8, 12, 24),
        topk: int = 2,
        mamba_layers: int = 2,
        mamba_d_state: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.topk = topk
        self.moe_windows = moe_windows
        # patch 
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == "end":
            self.pad = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1
        else:
            self.pad = None

        self.patch_embed = nn.Linear(patch_len, d_model)


        self.mamba_stack = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=mamba_d_state, dropout=dropout)
            for _ in range(mamba_layers)
        ])

        self.moe_attn = TopKMoEMultiWindowAttention(
            d_model=d_model, n_heads=n_heads,
            windows=moe_windows, topk=topk,
            attn_dropout=dropout, proj_dropout=dropout
        )


        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),                              # [B, L*D]
            nn.Linear(self.patch_num * d_model, pred_len * 2),
            nn.GELU(),
            nn.Linear(pred_len * 2, pred_len),
        )


        self.fr_patch = PatchContextGate(
            patch_num=self.patch_num,
            d_model=d_model,
            att_size=16,         
            mlp_hidden=128,      
            dropout=dropout,
            init_gate=-3.0      
        )

        self.adapt_mamba = MLPAdapter(d_model=d_model, widen=4, dropout=dropout, init_gate=-3.0)
        self.adapt_moe   = MLPAdapter(d_model=d_model, widen=4, dropout=dropout, init_gate=-3.0)  

    def forward(self, s_flat: torch.Tensor, x_mark=None, C=None) -> torch.Tensor:


        BC, L = s_flat.shape

        if self.pad is not None:
            s_flat = self.pad(s_flat)  # [B*C, I + stride]
        patches = s_flat.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B*C, P, K]
        z = self.patch_embed(patches)  # [B*C, P, D] 
        
        z = self.fr_patch(z)

        z_mamba = z.contiguous()
        for blk in self.mamba_stack:
            z_mamba = blk(z_mamba)     # [B*C, P, D]

        z_mamba = self.adapt_mamba(z_mamba)  

        z_moe = self.moe_attn(z)       
        z_moe = self.adapt_moe(z_moe)

        z_cat = torch.cat([z_mamba, z_moe], dim=-1)  # [B*C, P, 2D]
        z_fused = self.fuse(z_cat)                   # [B*C, P, D]

        out = self.head(z_fused)       # [B*C, pred_len]
        return out

class MultiWindowAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        windows: List[int] = (4, 8, 12),
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.windows = list(windows)  
        self.num_windows = len(windows)  

        self.attn_branches = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=attn_dropout,
                batch_first=True  # [B, L, D]
            ) for _ in self.windows
        ])

        self.branch_fusion = nn.Sequential(
            nn.Linear(d_model * self.num_windows, d_model), 
            nn.GELU(), 
            nn.Dropout(proj_dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, L, D = x.shape
        device = x.device

        branch_outputs = []
        for window_size, attn in zip(self.windows, self.attn_branches):
        
            local_mask = self._create_local_mask(L, window_size, device)
            attn_out, _ = attn(x, x, x, attn_mask=local_mask)  # [B, L, D]
            branch_outputs.append(attn_out)

        concat_outs = torch.cat(branch_outputs, dim=-1)  # [B, L, D * num_windows]
        fused_x = self.branch_fusion(concat_outs)        # [B, L, D]

        return fused_x

    def _create_local_mask(self, seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:

        mask = torch.full((seq_len, seq_len), False, device=device, dtype=torch.bool)
        for i in range(seq_len):
            visible_start = max(0, i - window_size)
            visible_end = min(seq_len, i + window_size + 1)  
            mask[i, :visible_start] = True
            mask[i, visible_end:] = True
        return mask


class TrendExtractor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 128,
        mamba_layers: int = 2,
        mamba_d_state: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len

        # 将标量序列扩为特征维度 D： [B*C, L, 1] -> [B*C, L, D]
        self.in_proj = nn.Linear(1, d_model)

        self.mamba_stack = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=mamba_d_state, dropout=dropout)
            for _ in range(mamba_layers)
        ])

        # 先把特征降回标量，再从 L -> pred_len 做时间维线性映射
        self.to_scalar = nn.Linear(d_model, 1)      # [B*C, L, D] -> [B*C, L, 1]
        self.time_proj = nn.Linear(seq_len, pred_len)  # [B*C, L] -> [B*C, pred_len]

    def forward(self, t_flat: torch.Tensor) -> torch.Tensor:
        """
        t_flat: [B*C, I] （通道独立）
        返回:   [B*C, pred_len]
        """
        x = t_flat.unsqueeze(-1)            # [B*C, L, 1]
        x = self.in_proj(x)                 # [B*C, L, D]
        x = x.contiguous()
        for blk in self.mamba_stack:
            x = blk(x)                      # [B*C, L, D]
        x = self.to_scalar(x).squeeze(-1)   # [B*C, L]
        out = self.time_proj(x)             # [B*C, pred_len]
        return out


class RCS1DTime(nn.Module):
    def __init__(self, dim, out_dim, num_heads=4, bias=True, attn_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.d_head = dim // num_heads

        # 温度初始化稍“钝化”，减小初始扰动
        self.temperature = nn.Parameter(torch.full((num_heads,1,1), 0.5))
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Conv2d(dim, dim*3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim*3, dim*3, kernel_size=(3,1),
                                stride=1, dilation=(1,1), padding=(1,0),
                                groups=dim*3, bias=bias)
        self.proj_out = nn.Conv2d(dim, out_dim, 1, bias=bias)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_map, res_map):
        B, D, P, _ = x_map.shape
        qkv = self.qkv_dw(self.qkv(x_map))
        q, k, _ = torch.chunk(qkv, 3, dim=1)
        v = res_map

        def split(t):
            t = t.squeeze(-1).view(B, self.num_heads, self.d_head, P)
            return t

        q = split(q); k = split(k); v = split(v)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)                  
        out  = attn @ v                              # [B, H, Dh, P]

        out = out.view(B, D, P).unsqueeze(-1)        # [B, D, P, 1]
        out = self.proj_out(out)
        out = (x_map + res_map) + self.gamma * out
        return out



import torch
import torch.nn as nn
import torch.nn.functional as F


class _MambaIEU_Patch(nn.Module):

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mamba(x)
        return y

class _SelfAttnIEU_Patch(nn.Module):

    def __init__(self, d_model: int, att_size: int = 16, dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(d_model, att_size)
        self.k = nn.Linear(d_model, att_size)
        self.v = nn.Linear(d_model, att_size)
        self.proj = nn.Linear(att_size, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*, P, D]
        q = self.q(x)   # [B*, P, A]
        k = self.k(x)   # [B*, P, A]
        v = self.v(x)   # [B*, P, A]

        A = q.size(-1)
        logits = torch.matmul(q, k.transpose(1, 2)) / (A ** 0.5)   # [B*, P, P]
        attn   = logits.softmax(dim=-1)
        attn   = self.drop(attn)
        y_a    = torch.matmul(attn, v)                              # [B*, P, A]
        y      = self.proj(y_a)                                     # [B*, P, D]
        return y


class _ContextMLP(nn.Module):

    def __init__(self, P: int, D: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        inp = P * D
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, D),
            nn.ReLU(inplace=True),  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bp, P, D = x.shape
        z = x.reshape(Bp, P * D)        # [B*, P*D]
        z = self.net(z)                 # [B*, D]
        return z.unsqueeze(1)           # [B*, 1, D]


class PatchContextGate(nn.Module):

    def __init__(self, patch_num: int, d_model: int,
                 att_size: int = 16, mlp_hidden: int = 128,
                 dropout: float = 0.0, init_gate: float = -3.0):
        super().__init__()
        self.P, self.D = patch_num, d_model

        # self.ieug_selfattn = _SelfAttnIEU_Patch(d_model, att_size, dropout)
        self.ieug_selfattn = _MambaIEU_Patch(d_model)
        self.ieug_ctx      = _ContextMLP(patch_num, d_model, mlp_hidden, dropout)

        # self.iew_selfattn  = _SelfAttnIEU_Patch(d_model, att_size, dropout)
        self.iew_selfattn = _MambaIEU_Patch(d_model)
        self.iew_ctx       = _ContextMLP(patch_num, d_model, mlp_hidden, dropout)
        self.w_proj        = nn.Linear(d_model, 1)

        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.tensor(init_gate))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        Bp, P, D = z.shape
        assert P == self.P and D == self.D, f"expect [*,{self.P},{self.D}], got {z.shape}"
        z_vec = self.ieug_selfattn(z)                   # [B*,P,D]
        bit   = self.ieug_ctx(z)                        # [B*,1,D]
        com   = z_vec * bit                             # [B*,P,D]

        z_vec_w = self.iew_selfattn(z)                  # [B*,P,D]
        bit_w   = self.iew_ctx(z)                       # [B*,1,D]
        w_feat  = z_vec_w * bit_w                       # [B*,P,D]
        W_logit = self.w_proj(w_feat)                   # [B*,P,1]
        W       = torch.sigmoid(W_logit)                # (0,1)

        z_gate  = z * W + com * (1.0 - W)               # [B*,P,D]
        g = torch.sigmoid(self.gamma)                   # (0,1) 小值
        out = z + g * self.drop(z_gate - z)             # 稳定接入

        return out


class MLPAdapter(nn.Module):

    def __init__(self, d_model: int, widen: int = 4, dropout: float = 0.0, init_gate: float = -2.5):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, widen * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(widen * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.gamma = nn.Parameter(torch.tensor(init_gate))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.ffn(z)
        g = torch.sigmoid(self.gamma)
        return z + g * (y - z)