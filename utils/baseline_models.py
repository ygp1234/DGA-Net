# utils/baseline_models.py
import torch
from models.unet import UNet
from models.unet_plus import UNetPlusPlus
from models.ra_unet import RAUNet
# ... 其他基线模型导入

def load_baseline_model(model_name, config):
    """加载基线模型，使用论文统一配置"""
    if model_name == "U-Net":
        return UNet(in_channels=1, out_channels=3, base_channels=64)
    elif model_name == "UNet++":
        return UNetPlusPlus(in_channels=1, out_channels=3)
    elif model_name == "RA-UNet":
        return RAUNet(in_channels=1, out_channels=3)
    # ... 其他模型的初始化逻辑
    else:
        raise ValueError(f"不支持的基线模型：{model_name}")