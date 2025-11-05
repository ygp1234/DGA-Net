import nibabel as nib
import numpy as np
from config.config import config


def load_nii_file(file_path):
    """加载NIfTI格式的CT图像或标签"""
    img = nib.load(file_path)
    data = img.get_fdata()
    affine = img.affine
    return data, affine


def clip_hu(data):
    """根据论文设置裁剪HU值范围"""
    data = np.clip(data, config.HU_RANGE[0], config.HU_RANGE[1])
    return data


def normalize_intensity(data):
    """全局归一化：(I - mean) / std"""
    data = (data - config.MEAN) / config.STD
    return data


def resample_spacing(data, original_affine, target_spacing):
    """重采样到目标间距（论文使用三阶样条插值）"""
    from scipy.ndimage import zoom

    # 计算缩放因子
    original_spacing = np.abs(original_affine[:3, :3]).max(axis=1)
    zoom_factors = original_spacing / target_spacing

    # 对图像和标签分别插值（标签用最近邻）
    if len(data.shape) == 4:  # 多通道图像
        resampled_data = np.zeros((data.shape[0],) + tuple(np.round(data.shape[1:] * zoom_factors).astype(int)))
        for i in range(data.shape[0]):
            resampled_data[i] = zoom(data[i], zoom_factors, order=3)
    else:  # 标签
        resampled_data = zoom(data, zoom_factors, order=0)

    return resampled_data


def extract_foreground(data):
    """提取前景区域（0.5%-99.5%分位数范围）"""
    mask = (data > np.percentile(data, 0.5)) & (data < np.percentile(data, 99.5))
    return data * mask


def preprocess_single_case(img_path, label_path=None):
    """单病例预处理流程：加载→HU裁剪→前景提取→归一化→重采样"""
    # 处理图像
    img_data, img_affine = load_nii_file(img_path)
    img_data = clip_hu(img_data)
    img_data = extract_foreground(img_data)
    img_data = normalize_intensity(img_data)
    img_data = resample_spacing(img_data, img_affine, config.TARGET_SPACING)

    # 处理标签（如果存在）
    label_data = None
    if label_path is not None:
        label_data, _ = load_nii_file(label_path)
        label_data = resample_spacing(label_data, img_affine, config.TARGET_SPACING)

    return img_data, label_data


def batch_preprocess(raw_data_dir, save_dir):
    """批量预处理数据集"""
    import os
    os.makedirs(save_dir, exist_ok=True)

    # 遍历原始数据集（LiTS2017格式：image_xxx.nii.gz, label_xxx.nii.gz）
    for case in os.listdir(raw_data_dir):
        if "image" in case:
            case_id = case.split("_")[-1].split(".")[0]
            img_path = os.path.join(raw_data_dir, case)
            label_path = os.path.join(raw_data_dir, f"label_{case_id}.nii.gz")

            img_data, label_data = preprocess_single_case(img_path, label_path)

            # 保存预处理后的数据
            np.savez(os.path.join(save_dir, f"case_{case_id}.npz"), img=img_data, label=label_data)


if __name__ == "__main__":
    # 执行批量预处理
    batch_preprocess(config.RAW_DATA_PATH, config.PROCESSED_DATA_PATH)