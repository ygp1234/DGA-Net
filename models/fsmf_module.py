import torch
import torch.nn as nn
import torch.fft as fft


class FourierSpectralLearningBlock(nn.Module):
    """傅里叶谱学习块：学习振幅和相位特征"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # 傅里叶变换：实部+虚部 -> 振幅+相位
        fft_x = fft.fft2(x, dim=(-2, -1))
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)

        # 合并振幅和相位，通过1x1卷积学习
        amp_phase = torch.cat([amp, phase], dim=1)
        amp_phase = self.leaky_relu(self.conv1x1(amp_phase))

        # 逆傅里叶变换
        amp_out = amp_phase[:, :amp_phase.shape[1] // 2, ...]
        phase_out = amp_phase[:, amp_phase.shape[1] // 2:, ...]
        fft_out = amp_out * torch.exp(1j * phase_out)
        x_out = fft.ifft2(fft_out, dim=(-2, -1)).real

        return x_out


class MultiScaleResidualBlock(nn.Module):
    """多尺度残差块：提取细节特征"""

    def __init__(self, channels):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # 三路径特征融合
        path1 = self.conv3x3(x)
        path2 = self.upsample(self.conv3x3(self.downsample(x)))
        path3 = self.conv3x3(self.upsample(self.downsample(self.downsample(x))))

        return path1 + path2 + path3


class FSMFModule(nn.Module):
    """Fourier Spectral-learning Multi-scale Fusion 模块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fourier_block = FourierSpectralLearningBlock(out_channels, out_channels)
        self.multi_scale_block = MultiScaleResidualBlock(out_channels)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.double_conv(x)
        fourier_feat = self.fourier_block(x)
        multi_scale_feat = self.multi_scale_block(x)
        out = fourier_feat + multi_scale_feat
        return self.maxpool(out), out  # 返回下采样后特征和跳连特征