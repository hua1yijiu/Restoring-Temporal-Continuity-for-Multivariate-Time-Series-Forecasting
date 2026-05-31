import torch
from torch import nn
from typing import List
from mamba_ssm import Mamba
from layers.newLayer import SeasonalityExtractor, TrendExtractor


class Network(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        padding_patch: str = "none",
        d_model: int = 128,
        n_heads: int = 4,
        moe_windows: List[int] = (4, 8, 12),
        topk: int = 2,
        mamba_layers_season: int = 1,
        mamba_layers_trend: int = 1,
        mamba_d_state: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()


        # self.season = SeasonalityExtractor(
        #     seq_len=seq_len, pred_len=pred_len,
        #     patch_len=patch_len, stride=stride, padding_patch=padding_patch,
        #     d_model=d_model, n_heads=n_heads,
        #     moe_windows=moe_windows, 
        #     mamba_layers=mamba_layers_season, mamba_d_state=mamba_d_state,
        #     dropout=dropout
        # )


        # Seasonality extractor: Patch + Parallel(Mamba | TopK-MoE) + Head
        self.season = SeasonalityExtractor(
            seq_len=seq_len, pred_len=pred_len,
            patch_len=patch_len, stride=stride, padding_patch=padding_patch,
            d_model=d_model, n_heads=n_heads,
            moe_windows=moe_windows, topk=topk,
            mamba_layers=mamba_layers_season, mamba_d_state=mamba_d_state,
            dropout=dropout
        )

        # Trend extractor: Mamba-only
        self.trend = TrendExtractor(
            seq_len=seq_len, pred_len=pred_len,
            d_model=d_model, mamba_layers=mamba_layers_trend,
            mamba_d_state=mamba_d_state, dropout=dropout
        )

        self.fuse_head = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s: torch.Tensor, t: torch.Tensor, batch_x_mark=None, time_feat_dim=None) -> torch.Tensor:
        """
        s: [B, L, C]  (seasonality)
        t: [B, L, C]  (trend)
        返回: [B, pred_len, C]
        """
        # B, L, C = s.shape
        # s = s.permute(0, 2, 1).reshape(B * C, L)  # [B*C, L]
        # t = t.permute(0, 2, 1).reshape(B * C, L)  # [B*C, L]

        s = s.permute(0, 2, 1)  # [B, C, L]
        t = t.permute(0, 2, 1)  # [B, C, L]

        B, C, L = s.shape
        s = s.reshape(B * C, L)
        t = t.reshape(B * C, L)


        s_out = self.season(s, batch_x_mark, C=C)  # [B*C, pred_len]
        t_out = self.trend(t)   # [B*C, pred_len]

        x = torch.cat([s_out, t_out], dim=-1)      # [B*C, 2*pred_len]
        x = self.fuse_head(x)                      # [B*C, pred_len]

        x = x.view(B, C, -1).permute(0, 2, 1)      # [B, pred_len, C]
        return x
