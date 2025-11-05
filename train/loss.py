import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice损失计算模块，基于论文中定义的DC loss公式实现
    用于衡量预测结果与标签之间的重叠程度，值越接近0表示分割效果越好
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # 避免分母为0的平滑项

    def forward(self, pred, target):
        """
        参数说明：
        - pred: 模型预测输出，形状为[B, C, H, W]，其中C为类别数（论文中C=3：背景0、肝脏1、肿瘤2）
        - target: 标签数据，形状为[B, H, W]，值为类别索引（0/1/2）

        返回值：
        - dice_loss: 针对肿瘤类别的Dice损失（论文中重点关注肿瘤分割性能）
        """
        # 仅计算肿瘤类别的Dice损失（类别索引为2）
        pred_tumor = F.softmax(pred, dim=1)[:, 2, :, :]  # 肿瘤类别概率图
        target_tumor = (target == 2).float()  # 肿瘤类别标签二值化

        # 计算交集和并集（按batch维度累加）
        intersection = (pred_tumor * target_tumor).sum(dim=[1, 2])
        union = pred_tumor.sum(dim=[1, 2]) + target_tumor.sum(dim=[1, 2])

        # 论文中DC loss公式：DC = 2*|PR∩GT| / (|PR| + |GT|)，损失取1 - DC
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()  # 对batch内所有样本求平均

        return dice_loss


class CrossEntropyLoss(nn.Module):
    """
    交叉熵损失模块，适配论文中多类别（背景、肝脏、肿瘤）分割任务
    用于衡量预测概率分布与标签之间的信息差异
    """

    def __init__(self, weight=None):
        super(CrossEntropyLoss, self).__init__()
        # 论文未指定类别权重，默认使用等权重；可根据数据类别不平衡情况调整
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred, target):
        """
        参数说明：
        - pred: 模型预测输出，形状为[B, C, H, W]
        - target: 标签数据，形状为[B, H, W]，值为类别索引（0/1/2）

        返回值：
        - ce_loss: 多类别交叉熵损失
        """
        return self.ce_loss(pred, target.long())  # 确保标签为长整型（类别索引）


class MixedLoss(nn.Module):
    """
    混合损失函数：论文中定义的CE loss与DC loss加权求和
    结合两种损失的优势，既关注类别概率分布差异，又关注分割区域重叠度
    """

    def __init__(self, alpha=0.5):
        super(MixedLoss, self).__init__()
        self.alpha = alpha  # CE损失权重，DC损失权重为1 - alpha
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        """
        参数说明：
        - pred: 模型预测输出，形状为[B, C, H, W]
        - target: 标签数据，形状为[B, H, W]，值为类别索引（0/1/2）

        返回值：
        - mixed_loss: 混合损失值，计算公式为 alpha*CE + (1-alpha)*DC
        """
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        mixed_loss = self.alpha * ce + (1 - self.alpha) * dice
        return mixed_loss