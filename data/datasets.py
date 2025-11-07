import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config.config import config


class LiTSDataset(Dataset):
    """
    针对LiTS2017数据集的加载类，适配论文中肝脏肿瘤分割任务的数据读取与预处理后数据加载
    支持训练/验证/测试集划分，完全对齐论文3.1节（Datasets）中LiTS2017数据集的使用规范
    """

    def __init__(self, data_dir, split="train", transform=None):
        """
        参数说明：
        - data_dir: 预处理后数据的存储目录（对应论文中"processed data"，由preprocess.py生成）
        - split: 数据集划分类型，可选"train"/"val"/"test"，对应论文中7:1:2的划分比例
        - transform: 数据增强变换（可选，论文未明确提及额外增强，默认None）
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.case_files = self._get_case_files()  # 获取对应划分的病例文件列表
        self._validate_case_labels()  # 验证标签格式符合论文定义

    def _get_case_files(self):
        """
        根据论文3.1节的划分比例（训练:验证:测试=7:1:2）筛选病例文件
        读取预处理后的数据文件（.npz格式，包含"img"和"label"字段）
        """
        # 获取所有预处理后的病例文件
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith(".npz")]
        total_cases = len(all_files)

        # 按论文比例划分索引
        train_end = int(total_cases * config.TRAIN_TEST_SPLIT[0])
        val_end = train_end + int(total_cases * config.TRAIN_TEST_SPLIT[1])

        # 分配对应划分的文件
        if self.split == "train":
            case_files = all_files[:train_end]
        elif self.split == "val":
            case_files = all_files[train_end:val_end]
        elif self.split == "test":
            case_files = all_files[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}, must be 'train'/'val'/'test'")

        # 拼接完整文件路径
        case_files = [os.path.join(self.data_dir, f) for f in case_files]
        return case_files

    def _validate_case_labels(self):
        """
        验证标签格式符合论文3.1节定义：
        - 标签值仅包含0（背景）、1（肝脏）、2（肿瘤）三类
        - 确保每个病例至少包含肝脏区域（论文中LiTS2017训练集131例均含肝脏）
        """
        for file_path in self.case_files[:5]:  # 抽样验证前5个病例
            data = np.load(file_path)
            label = data["label"]
            unique_labels = np.unique(label)

            # 检查标签值是否合法
            if not set(unique_labels).issubset({0, 1, 2}):
                raise ValueError(f"Invalid label values {unique_labels} in {file_path}, must be 0/1/2")

            # 检查是否包含肝脏区域（标签1）
            if 1 not in unique_labels:
                raise ValueError(f"Case {file_path} missing liver region (label=1),不符合LiTS2017数据集规范")

    def __getitem__(self, idx):
        """
        读取单病例数据，返回模型输入格式（图像+标签）
        输出格式对齐论文中模型输入要求：CT图像为单通道，标签为类别索引
        """
        file_path = self.case_files[idx]
        data = np.load(file_path)
        img = data["img"]  # 预处理后的CT图像，形状为[H, W]（单切片）或[D, H, W]（体积）
        label = data["label"]  # 对应标签，形状与图像一致

        # 处理图像维度：确保为[C, H, W]（单通道），适配2D模型输入（论文中编码器基于2D U-Net结构）
        if len(img.shape) == 2:  # 单切片（H, W）
            img = np.expand_dims(img, axis=0)  # 增加通道维度→[1, H, W]
        elif len(img.shape) == 3:  # 体积（D, H, W）：按论文默认取中间切片（简化处理，或可改为切片遍历）
            mid_slice = img.shape[0] // 2
            img = img[mid_slice]
            label = label[mid_slice]
            img = np.expand_dims(img, axis=0)  # 增加通道维度→[1, H, W]
        else:
            raise ValueError(f"Invalid image shape {img.shape}, must be 2D (H,W) or 3D (D,H,W)")

        # 处理标签维度：确保为[H, W]（去除通道维度）
        if len(label.shape) == 3:
            label = label[mid_slice]  # 与图像切片对应

        # 数据类型转换：适配PyTorch模型（float32图像，int64标签）
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        # 应用数据增强（若有）
        if self.transform is not None:
            img, label = self.transform(img, label)

        return img, label

    def __len__(self):
        """返回数据集总病例数"""
        return len(self.case_files)

def filter_by_tumor_size(self, min_diameter=0, max_diameter=np.inf):
    """按肿瘤最大直径过滤数据集（单位：cm）"""
    filtered_samples = []
    for sample in self.samples:
        tumor_mask = sample["label"] == 2  # 肿瘤标签=2
        if not np.any(tumor_mask):
            continue
        # 计算肿瘤最大直径（像素转cm，需结合CT图像间距）
        tumor_coords = np.where(tumor_mask)
        diameter_pixel = np.max(tumor_coords) - np.min(tumor_coords)
        diameter_cm = diameter_pixel * self.spacing[0] / 10  # 假设spacing单位为mm，转cm
        if min_diameter <= diameter_cm < max_diameter:
            filtered_samples.append(sample)
    return LiTSDataset(samples=filtered_samples, **self.kwargs)  # 返回新数据集实例
class ThreeDIRCADbDataset(Dataset):
    """
    针对3DIRCADb数据集的加载类，适配论文4.3节（Experiment on the 3Dircadb dataset）的验证需求
    数据集规范遵循论文描述：20例CT扫描（10男10女），15例含肝脏肿瘤，标签格式与LiTS2017一致
    """

    def __init__(self, data_dir, split="test", transform=None):
        """
        参数说明：
        - data_dir: 预处理后3DIRCADb数据的存储目录
        - split: 数据集划分类型，仅支持"test"（论文中用于模型泛化性验证，不划分训练集）
        - transform: 数据增强变换（默认None，验证阶段不使用增强）
        """
        if split != "test":
            raise ValueError(f"3DIRCADb仅用于验证，split必须为'test'，当前输入{split}")

        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.case_files = self._get_test_files()  # 获取所有测试病例
        self._validate_tumor_cases()  # 验证含肿瘤病例数量符合论文描述

    def _get_test_files(self):
        """获取3DIRCADb所有预处理后的测试文件（论文中20例均用于验证）"""
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith(".npz")]
        case_files = [os.path.join(self.data_dir, f) for f in all_files]

        # 检查数据集总量是否为20例（论文中3DIRCADb含20例CT扫描）
        if len(case_files) != 20:
            raise ValueError(f"3DIRCADb数据集应含20例，当前{len(case_files)}例，不符合论文规范")

        return case_files

    def _validate_tumor_cases(self):
        """验证含肿瘤病例数量符合论文4.3节描述：15例含肝脏肿瘤"""
        tumor_case_count = 0
        for file_path in self.case_files:
            data = np.load(file_path)
            label = data["label"]
            if 2 in np.unique(label):  # 检查是否含肿瘤（标签2）
                tumor_case_count += 1

        if tumor_case_count != 15:
            raise ValueError(f"3DIRCADb应含15例肿瘤病例，当前{tumor_case_count}例，不符合论文规范")

    def __getitem__(self, idx):
        """读取单病例数据，逻辑与LiTSDataset一致，确保与模型输入格式兼容"""
        file_path = self.case_files[idx]
        data = np.load(file_path)
        img = data["img"]
        label = data["label"]

        # 维度处理：与LiTSDataset一致，确保[C, H, W]图像和[H, W]标签
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        elif len(img.shape) == 3:
            mid_slice = img.shape[0] // 2
            img = img[mid_slice]
            label = label[mid_slice]
            img = np.expand_dims(img, axis=0)

        # 数据类型转换
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        # 应用变换（验证阶段可选）
        if self.transform is not None:
            img, label = self.transform(img, label)

        return img, label

    def __len__(self):
        return len(self.case_files)


# 数据集工厂函数：根据配置自动选择LiTS2017或3DIRCADb
def get_dataset(data_dir, dataset_name, split="train", transform=None):
    """
    参数说明：
    - dataset_name: 数据集名称，可选"LiTS2017"/"3DIRCADb"（论文中仅使用这两个数据集）
    - 其他参数同上述数据集类
    """
    if dataset_name == "LiTS2017":
        return LiTSDataset(data_dir, split, transform)
    elif dataset_name == "3DIRCADb":
        return ThreeDIRCADbDataset(data_dir, split, transform)
    else:
        raise ValueError(f"仅支持LiTS2017/3DIRCADb数据集，当前输入{dataset_name}")