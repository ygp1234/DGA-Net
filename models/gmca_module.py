import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadInteractiveAttention(nn.Module):
    """多头交互注意力块"""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, q, k, v):
        B, C, H, W = q.shape
        # reshape到注意力维度
        q = q.permute(0, 2, 3, 1).reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.permute(0, 2, 3, 1).reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.permute(0, 2, 3, 1).reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        # 输出投影
        out = out.transpose(1, 2).reshape(B, H * W, C).permute(0, 2, 1).reshape(B, C, H, W)
        out = self.out_proj(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


class FourBranchFeatureAggregation(nn.Module):
    """四分支特征聚合块（不同膨胀率卷积）"""

    def __init__(self, channels):
        super().__init__()
        self.dil_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.dil_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.dil_conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=5, dilation=5)
        self.dil_conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=7, dilation=7)
        self.bn = nn.BatchNorm2d(channels * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat1 = self.dil_conv1(x)
        feat2 = self.dil_conv2(x)
        feat3 = self.dil_conv3(x)
        feat4 = self.dil_conv4(x)
        out = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        out = self.relu(self.bn(out))
        return out


class GMCAModule(nn.Module):
    """Group Multi-head Cross-Attention Aggregation 模块"""

    def __init__(self, in_channels, out_channels, num_heads=4, num_groups=4):
        super().__init__()
        self.num_groups = num_groups
        self.group_split = in_channels // num_groups
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layer_norm = nn.LayerNorm(in_channels)
        self.mia = MultiHeadInteractiveAttention(in_channels, num_heads)
        self.fbfa = FourBranchFeatureAggregation(in_channels)
        self.fusion_conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)

    def forward(self, fsmf_feat, maha_feat, prev_feat):
        # 上采样前一层特征
        prev_feat = self.upsample(prev_feat)
        # 分组并归一化
        B, C, H, W = fsmf_feat.shape
        grouped_feats = []
        for g in range(self.num_groups):
            start = g * self.group_split
            end = (g + 1) * self.group_split
            fsmf_g = fsmf_feat[:, start:end, ...]
            maha_g = maha_feat[:, start:end, ...]
            prev_g = prev_feat[:, start:end, ...]

            # 合并并归一化
            fused_g = torch.cat([fsmf_g, maha_g, prev_g], dim=1)
            fused_g = self.layer_norm(fused_g.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            grouped_feats.append(fused_g)

        # 多头交互注意力
        mia_outs = [self.mia(gf[:, :C // 3, ...], gf[:, C // 3:2 * C // 3, ...], gf[:, 2 * C // 3:, ...]) for gf in
                    grouped_feats]
        mia_out = torch.cat(mia_outs, dim=1)

        # 四分支特征聚合
        fbfa_out = self.fbfa(mia_out)

        # 特征融合
        out = self.fusion_conv(fbfa_out)
        return out