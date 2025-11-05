import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_coefficient(pred: np.ndarray, target: np.ndarray, class_id: int = 2) -> float:
    """
    计算指定类别的Dice系数，对应论文3.3节中Dice Score（DCS）定义
    用于衡量预测区域与真实区域的重叠程度，值范围[0,1]，越接近1表示分割效果越好

    参数：
    - pred: 模型预测结果，numpy数组，值为类别索引（0/1/2）
    - target: 真实标签，numpy数组，值为类别索引（0/1/2）
    - class_id: 目标类别ID，论文中重点关注肿瘤类别（默认2）

    返回：
    - dice: 目标类别的Dice系数
    """
    # 提取目标类别的二值掩码
    pred_mask = (pred == class_id).astype(np.float32)
    target_mask = (target == class_id).astype(np.float32)

    # 计算交集与并集（避免除以零）
    intersection = np.sum(pred_mask * target_mask)
    union = np.sum(pred_mask) + np.sum(target_mask)
    if union < 1e-6:  # 若目标区域为空，Dice设为1（无误差）
        return 1.0
    return (2.0 * intersection) / union


def volume_overlap_error(pred: np.ndarray, target: np.ndarray, class_id: int = 2) -> float:
    """
    计算体积重叠误差（VOE），对应论文3.3节公式定义
    衡量预测体积与真实体积的重叠偏差，值范围[0,1]，越接近0表示重叠越好

    参数同dice_coefficient，返回VOE值（百分比形式已转换为小数）
    """
    pred_mask = (pred == class_id).astype(np.float32)
    target_mask = (target == class_id).astype(np.float32)

    intersection = np.sum(pred_mask * target_mask)
    union = np.sum(pred_mask) + np.sum(target_mask) - intersection
    if union < 1e-6:
        return 0.0
    return 1.0 - (intersection / union)


def relative_volume_difference(pred: np.ndarray, target: np.ndarray, class_id: int = 2) -> float:
    """
    计算相对体积差异（RVD），对应论文3.3节公式定义
    衡量预测体积与真实体积的比例差异，值越接近0表示体积偏差越小

    参数同dice_coefficient，返回RVD值
    """
    pred_volume = np.sum(pred == class_id)
    target_volume = np.sum(target == class_id)

    if target_volume < 1e-6:  # 真实体积为空时，RVD设为0
        return 0.0
    return (pred_volume - target_volume) / target_volume


def average_symmetric_surface_distance(pred: np.ndarray, target: np.ndarray, class_id: int = 2) -> float:
    """
    计算平均对称表面距离（ASSD），对应论文3.3节定义
    衡量预测区域与真实区域表面的平均距离，单位与图像分辨率一致，值越小表示边界越接近

    参数同dice_coefficient，返回ASSD值（mm）
    """
    pred_mask = (pred == class_id).astype(np.bool_)
    target_mask = (target == class_id).astype(np.bool_)

    # 若任一掩码为空，返回0（无表面可计算）
    if not np.any(pred_mask) or not np.any(target_mask):
        return 0.0

    # 计算欧氏距离变换
    pred_dist = distance_transform_edt(~pred_mask)
    target_dist = distance_transform_edt(~target_mask)

    # 提取表面像素并计算平均距离
    pred_surface = pred_mask & (pred_dist == 1)
    target_surface = target_mask & (target_dist == 1)

    avg_dist_pred = np.mean(pred_dist[target_surface])
    avg_dist_target = np.mean(target_dist[pred_surface])

    return (avg_dist_pred + avg_dist_target) / 2.0


def compute_all_metrics(preds: list[np.ndarray], targets: list[np.ndarray], class_id: int = 2) -> dict:
    """
    批量计算论文3.3节中所有评估指标（DPC、DG、VOE、RVD、ASSD）
    适配多病例批量评估场景，与论文4.1节（对比实验）和4.2节（消融实验）的指标输出一致

    参数：
    - preds: 预测结果列表，每个元素为单病例numpy数组
    - targets: 真实标签列表，每个元素为单病例numpy数组
    - class_id: 目标类别ID（默认肿瘤类别2）

    返回：
    - metrics: 包含所有指标的字典，键为指标名（与论文一致）
    """
    # 计算Dice per case（DPC）：单病例Dice平均值
    dpc_list = [dice_coefficient(p, t, class_id) for p, t in zip(preds, targets)]
    dpc = np.mean(dpc_list)

    # 计算Global Dice（DG）：所有病例合并后的Dice
    all_pred = np.concatenate([p.flatten() for p in preds])
    all_target = np.concatenate([t.flatten() for t in targets])
    dg = dice_coefficient(all_pred, all_target, class_id)

    # 计算其他指标（全局）
    voe = volume_overlap_error(all_pred, all_target, class_id)
    rvd = relative_volume_difference(all_pred, all_target, class_id)
    assd = average_symmetric_surface_distance(all_pred, all_target, class_id)

    return {
        "DPC": round(dpc, 4),
        "DG": round(dg, 4),
        "VOE": round(voe, 4),
        "RAVD": round(rvd, 4),  # 论文中部分表格写为RAVD，与RVD同义
        "ASSD": round(assd, 4)
    }