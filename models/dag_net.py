import torch
import torch.nn as nn
from models.fsmf_module import FSMFModule
from models.maha_module import MAHAModule
from models.gmca_module import GMCAModule


class DAGNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.IN_CHANNELS
        self.num_classes = config.NUM_CLASSES
        self.depth = config.ENCODER_DEPTH
        self.channels = [64, 128, 256, 512, 1024]  # 各层通道数

        # 编码器：双分支（FSMF + MAHA）
        self.fsmf_encoders = nn.ModuleList()
        self.maha_encoders = nn.ModuleList()
        for i in range(self.depth):
            in_ch = self.in_channels if i == 0 else self.channels[i - 1]
            out_ch = self.channels[i]
            self.fsmf_encoders.append(FSMFModule(in_ch, out_ch))
            self.maha_encoders.append(MAHAModule(in_ch, out_ch))

        # 解码器：GMCA模块
        self.gmca_decoders = nn.ModuleList()
        for i in range(self.depth - 1, 0, -1):
            in_ch = self.channels[i] * 2  # 双分支特征合并
            out_ch = self.channels[i - 1]
            self.gmca_decoders.append(GMCAModule(in_ch, out_ch))

        # 最终输出层
        self.final_conv = nn.Conv2d(self.channels[0], self.num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器前向传播（保存跳连特征）
        fsmf_skips = []
        maha_skips = []
        fsmf_feat = x
        maha_feat = x

        for fsmf_enc, maha_enc in zip(self.fsmf_encoders, self.maha_encoders):
            fsmf_feat, fsmf_skip = fsmf_enc(fsmf_feat)
            maha_feat, maha_skip = maha_enc(maha_feat)
            fsmf_skips.append(fsmf_skip)
            maha_skips.append(maha_skip)

        # 解码器前向传播（融合双分支跳连特征）
        dec_feat = torch.cat([fsmf_feat, maha_feat], dim=1)
        for i, gmca_dec in enumerate(self.gmca_decoders):
            skip_idx = self.depth - 2 - i
            fsmf_skip = fsmf_skips[skip_idx]
            maha_skip = maha_skips[skip_idx]
            dec_feat = gmca_dec(fsmf_skip, maha_skip, dec_feat)

        # 最终输出
        out = self.final_conv(dec_feat)
        return out