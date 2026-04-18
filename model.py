"""
model.py - ConvNeXt-T + Snippet 时序聚合 + Attention MIL + 双头分类

架构:
  1. ConvNeXt-Tiny 提取单帧特征
  2. Snippet 内 1D temporal conv 聚合
  3. Attention MIL 聚合所有 snippet -> clip 级特征
  4. 双头输出: is_bidirectional (2类), lane_count (3类: 1/2/2+)
"""

import torch
import torch.nn as nn
import timm


class SnippetEncoder(nn.Module):
    """ConvNeXt-T backbone + snippet 内时序聚合."""

    def __init__(self, backbone_name="convnext_tiny", pretrained=True, feature_dim=768):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        backbone_dim = self.backbone.num_features
        self.feature_dim = feature_dim

        # 对齐维度（backbone 输出可能不等于 feature_dim）
        self.proj = nn.Identity() if backbone_dim == feature_dim else nn.Linear(backbone_dim, feature_dim)

        # snippet 内 1D temporal conv 聚合
        self.temporal_agg = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - T 帧组成的 snippet
        Returns:
            snippet_feat: (B, D)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.backbone(x)          # (B*T, backbone_dim)
        feat = self.proj(feat)            # (B*T, D)
        feat = feat.view(B, T, -1)       # (B, T, D)
        feat = feat.permute(0, 2, 1)     # (B, D, T)
        snippet_feat = self.temporal_agg(feat).squeeze(-1)  # (B, D)
        return snippet_feat


class AttentionMIL(nn.Module):
    """Gated Attention MIL 聚合 snippet -> clip."""

    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.attn_V = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, D) - N 个 snippet 特征
            mask: (B, N) - True 表示有效 snippet，False 表示 padding
        Returns:
            aggregated: (B, D)
            attn_weights: (B, N) - 注意力权重（可用于证据定位）
        """
        v = self.attn_V(x)   # (B, N, H)
        u = self.attn_U(x)   # (B, N, H)
        scores = self.attn_w(v * u)  # (B, N, 1)

        if mask is None:
            attn_weights = torch.softmax(scores, dim=1)  # (B, N, 1)
        else:
            if mask.dim() != 2 or mask.shape != x.shape[:2]:
                raise ValueError(f"mask shape must be (B, N) = {x.shape[:2]}, got {tuple(mask.shape)}")
            mask = mask.to(dtype=torch.bool, device=x.device).unsqueeze(-1)  # (B, N, 1)

            # 屏蔽 padding snippet；随后再乘 mask + 重归一化，确保全无效时不会出现 NaN
            scores = scores.masked_fill(~mask, -1e9)
            attn_weights = torch.softmax(scores, dim=1)
            attn_weights = attn_weights * mask
            denom = attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            attn_weights = attn_weights / denom

        aggregated = (x * attn_weights).sum(dim=1)    # (B, D)
        return aggregated, attn_weights.squeeze(-1)


class LaneMVPModel(nn.Module):
    """完整模型: ConvNeXt-T + Snippet聚合 + Attention MIL + 双头."""

    def __init__(self, cfg):
        super().__init__()
        mcfg = cfg["model"]
        self.snippet_length = mcfg["snippet_length"]

        self.snippet_encoder = SnippetEncoder(
            backbone_name=mcfg["backbone"],
            pretrained=mcfg["pretrained"],
            feature_dim=mcfg["feature_dim"],
        )
        self.attention_mil = AttentionMIL(
            feature_dim=mcfg["feature_dim"],
            hidden_dim=mcfg["mil_hidden_dim"],
        )
        self.dropout = nn.Dropout(mcfg["dropout"])

        # 双头分类
        self.direction_head = nn.Linear(mcfg["feature_dim"], mcfg["num_direction_classes"])
        self.lane_count_head = nn.Linear(mcfg["feature_dim"], mcfg["num_lane_classes"])

    def forward(self, snippets, masks=None):
        """
        Args:
            snippets: (B, N, T, C, H, W) - B个clip, 每个N个snippet, 每个T帧
            masks: (B, N) - True 表示有效 snippet，False 表示 padding
        Returns:
            dict with direction_logits, lane_count_logits, attention_weights
        """
        B, N, T, C, H, W = snippets.shape
        if masks is None:
            masks = torch.ones((B, N), dtype=torch.bool, device=snippets.device)
        else:
            masks = masks.to(device=snippets.device, dtype=torch.bool)

        # 编码每个 snippet
        snippets_flat = snippets.view(B * N, T, C, H, W)
        snippet_feats = self.snippet_encoder(snippets_flat)  # (B*N, D)
        snippet_feats = snippet_feats.view(B, N, -1)         # (B, N, D)

        # MIL 聚合
        clip_feat, attn_weights = self.attention_mil(snippet_feats, masks)  # (B, D), (B, N)
        clip_feat = self.dropout(clip_feat)

        # 双头输出
        direction_logits = self.direction_head(clip_feat)      # (B, 2)
        lane_count_logits = self.lane_count_head(clip_feat)    # (B, num_lane_classes)

        return {
            "direction_logits": direction_logits,
            "lane_count_logits": lane_count_logits,
            "attention_weights": attn_weights,
        }


def build_model(cfg):
    """构建模型的工厂函数."""
    return LaneMVPModel(cfg)
