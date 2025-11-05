import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAxisHadamardAttention(nn.Module):
    """多轴哈达玛注意力块"""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.mlp = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.head_dim, self.head_dim)
        )
        self.relative_pos_emb = nn.Parameter(torch.randn(3, self.head_dim))  # 三个轴的位置编码

    def forward(self, x):
        B, C, H, W = x.shape
        # 分块并reshape到注意力维度
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)  # B, num_heads, HW, head_dim

        # 三个轴的哈达玛注意力计算
        axes = [(2, 3), (1, 3), (1, 2)]  # 高-宽、通道-高、通道-宽
        attn_outputs = []
        for i, (axis1, axis2) in enumerate(axes):
            q = x.mean(dim=axis1, keepdim=True)
            k = x.mean(dim=axis2, keepdim=True)
            hadamard_attn = q * k  # 哈达玛积
            hadamard_attn = self.mlp(hadamard_attn)
            hadamard_attn = F.softmax(hadamard_attn + self.relative_pos_emb[i:i + 1], dim=-1)
            attn_outputs.append(x * hadamard_attn)

        # 融合三个轴的结果
        out = torch.cat(attn_outputs, dim=-1)
        out = out.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


class MAHAModule(nn.Module):
    """Multi-axis Aggregation Hadamard Attention 模块"""

    def __init__(self, in_channels, out_channels, num_groups=4):
        super().__init__()
        self.num_groups = num_groups
        self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=num_groups)
        self.attention = MultiAxisHadamardAttention(out_channels)
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.group_conv(x)
        attn_feat = self.attention(x)
        out = self.double_conv(attn_feat + x)  # 残差连接
        return self.maxpool(out), out  # 返回下采样后特征和跳连特征