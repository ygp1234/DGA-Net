import argparse
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy import stats
from config.config import config  # 复用项目配置（数据路径、超参数等）
from data.datasets import get_dataset  # 复用数据集加载逻辑
from models.dag_net import DAGNet, DGANetAblation  # 含消融变体的DGA-Net模型
from train.train import train_epoch, val_epoch  # 复用训练/验证逻辑
from train.loss import MixedLoss  # 复用CE+Dice混合损失（论文3.5节）
from evaluate.evaluate import evaluate_model  # 复用评估逻辑
from utils.logger import Logger  # 复用日志记录
from utils.metrics import compute_all_metrics  # 复用指标计算（DPC/DG/VOE等）


def parse_args():
    """解析命令行参数，支持指定消融模块、任务类型（肝/肿瘤）和目标表格"""
    parser = argparse.ArgumentParser(description="DGA-Net 消融实验脚本（复现表3/4/5/6）")
    parser.add_argument("--ablation-target", type=str, required=True,
                        choices=["table3", "table4", "table5", "table6"],
                        help="消融目标表格：table3(肝模块消融)、table4(肿瘤模块消融)、table5(肿瘤大小影响)、table6(高斯噪声影响)")
    parser.add_argument("--save-dir", type=str, default="./ablation_results",
                        help="消融结果（表格CSV）保存目录")
    parser.add_argument("--model-save-path", type=str, default="./ablation_checkpoints",
                        help="消融模型权重保存路径")
    parser.add_argument("--num-repeats", type=int, default=3,
                        help="实验重复次数（论文要求3次取平均，确保稳定性）")
    return parser.parse_args()


def get_ablation_variants():
    """定义所有消融变体（对应表3/4的6种消融情况，论文4.4.1节）"""
    return {
        "w/o FSMF": DGANetAblation(ablate="fsmf"),  # 移除FSMF模块
        "w/o ConvFFT": DGANetAblation(ablate="convfft"),  # 移除ConvFFT模块
        "w/o GMCA": DGANetAblation(ablate="gmca"),  # 移除GMCA模块
        "w/o MAHA": DGANetAblation(ablate="maha"),  # 移除MAHA模块
        "w/o One-Branch": DGANetAblation(ablate="one_branch"),  # 移除单分支（仅保留一个分支）
        "w/o Add(concat)": DGANetAblation(ablate="add_concat"),  # 移除像素加法融合（仅用拼接）
        "Ours": DAGNet(config)  # 完整DGA-Net（基准模型）
    }


def train_ablation_model(model, model_name, task, args):
    """训练单个消融变体模型，完全对齐论文实验配置（4.1节）"""
    logger = Logger(f"ablation_{model_name}_{task}")
    logger.log({"event": "ablation_train_start", "info": f"训练消融变体：{model_name}，任务：{task}"})

    # 加载LiTS2017数据集（论文消融实验仅用LiTS2017）
    train_dataset = get_dataset(config.PROCESSED_DATA_PATH, "LiTS2017", split="train")
    val_dataset = get_dataset(config.PROCESSED_DATA_PATH, "LiTS2017", split="val")
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 模型初始化（设备、优化器、学习率调度器完全匹配论文）
    model = model.to(config.DEVICE)
    criterion = MixedLoss(alpha=0.5)  # CE+Dice混合损失
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,  # 论文4.1节：初始学习率0.01
        momentum=0.9,
        weight_decay=(1 - config.CURRENT_EPOCH / config.MAX_EPOCHS) ** 0.9  # 多项式权重衰减
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: (1 - epoch / config.MAX_EPOCHS) ** 0.9
    )

    # 训练循环（论文4.1节：max_epoch=1000）
    best_val_loss = float("inf")
    for epoch in range(1, config.MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, _, _ = val_epoch(model, val_loader, criterion, config.DEVICE)

        # 保存最优模型（按验证损失）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(
                args.model_save_path, f"{model_name}_{task}_best.pth"
            )
            torch.save(model.state_dict(), model_save_path)

        lr_scheduler.step()
        logger.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    # 加载最优模型用于测试评估
    model.load_state_dict(torch.load(model_save_path))
    logger.log({"event": "ablation_train_end", "info": f"消融变体{model_name}训练完成"})
    logger.close()
    return model


def evaluate_ablation_model(model, model_name, task, args):
    """评估单个消融变体模型，返回含标准差和95%CI的指标（匹配论文表格格式）"""
    # 加载LiTS2017测试集（论文4.1节：测试集26例）
    test_dataset = get_dataset(config.PROCESSED_DATA_PATH, "LiTS2017", split="test")
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 确定评估类别ID（肝=1，肿瘤=2，论文4.1节标签定义）
    class_id = 1 if task == "liver" else 2

    # 多次重复评估（论文要求3次，计算标准差和95%CI）
    metrics_list = []
    for _ in range(args.num_repeats):
        metrics = evaluate_model(model, test_loader, config.DEVICE, class_id=class_id)
        metrics_list.append(metrics)

    # 计算均值、标准差、95%CI（置信区间）
    result = {"Method": model_name}
    for metric in ["DPC", "DG", "VOE", "RAVD", "ASSD"]:
        # 提取所有重复实验的该指标值
        metric_vals = [m[metric]["mean"] for m in metrics_list]
        mean_val = np.mean(metric_vals)
        std_val = np.std(metric_vals, ddof=1)  # 样本标准差
        # 95%CI计算（t分布，自由度=num_repeats-1）
        ci_interval = stats.t.interval(
            0.95, len(metric_vals) - 1, loc=mean_val, scale=stats.sem(metric_vals)
        )
        ci_lower, ci_upper = ci_interval

        # 格式化输出（匹配论文表格：值(标准差)[95%CI下限,95%CI上限]）
        if metric in ["DPC", "DG", "VOE", "RAVD"]:
            result[metric] = f"{mean_val:.4f}({std_val:.4f})\n[{ci_lower:.4f},{ci_upper:.4f}]"
        else:  # ASSD单位为mm，保留2位小数
            result[metric] = f"{mean_val:.2f}({std_val:.2f})\n[{ci_lower:.2f},{ci_upper:.2f}]"

    return result


def reproduce_table3_table4(args, task, target_table):
    """复现表3（肝模块消融）或表4（肿瘤模块消融），论文4.4.1节"""
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.model_save_path, exist_ok=True)
    logger = Logger(f"reproduce_{target_table}")
    logger.log({"event": "table_reproduce_start", "info": f"开始复现{target_table}（任务：{task}）"})

    # 获取所有消融变体
    ablation_variants = get_ablation_variants()
    table_data = []

    # 遍历所有消融变体，训练并评估
    for model_name, model in ablation_variants.items():
        # 训练消融模型
        trained_model = train_ablation_model(model, model_name, task, args)
        # 评估消融模型
        ablation_result = evaluate_ablation_model(trained_model, model_name, task, args)
        table_data.append(ablation_result)

    # 保存为CSV（可直接导入论文）
    df = pd.DataFrame(table_data)
    save_path = os.path.join(args.save_dir, f"{target_table}_Ablation_Results.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    # 日志与打印
    logger.log({"event": "table_reproduce_end", "info": f"{target_table}复现完成，保存路径：{save_path}"})
    logger.close()
    print(f"\n{target_table}（{task}）复现结果预览：")
    print(df.to_string(index=False))
    return save_path


def reproduce_table5(args):
    """复现表5（肿瘤大小影响），论文4.4.2节：按肿瘤最大直径分层（<2cm/2-5cm/>5cm）"""
    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger("reproduce_table5")
    logger.log({"event": "table5_reproduce_start", "info": "开始复现表5（肿瘤大小影响）"})

    # 1. 加载完整DGA-Net模型（仅评估基准模型，无需消融变体）
    model = DAGNet(config)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH))  # 加载论文训练好的最优模型
    model = model.to(config.DEVICE)

    # 2. 加载LiTS2017测试集并按肿瘤大小分层
    test_dataset = get_dataset(config.PROCESSED_DATA_PATH, "LiTS2017", split="test")
    # 按肿瘤最大直径分层（论文4.4.2节临床标准）
    small_tumors = test_dataset.filter_by_tumor_size(max_diameter=2)  # <2cm
    medium_tumors = test_dataset.filter_by_tumor_size(min_diameter=2, max_diameter=5)  # 2-5cm
    large_tumors = test_dataset.filter_by_tumor_size(min_diameter=5)  # >5cm
    size_strata = {
        "Small Tumors (<2 cm)": small_tumors,
        "Medium Tumors (2–5 cm)": medium_tumors,
        "Large Tumors (>5 cm)": large_tumors
    }

    # 3. 定义需对比的基线模型（论文表5中的8个SOTA）
    baseline_models = {
        "UNet": torch.load(config.BASELINE_MODEL_PATHS["UNet"]),
        "UNet++": torch.load(config.BASELINE_MODEL_PATHS["UNet++"]),
        "RA-UNet": torch.load(config.BASELINE_MODEL_PATHS["RA-UNet"]),
        "nnUNet": torch.load(config.BASELINE_MODEL_PATHS["nnUNet"]),
        "EGE-UNet": torch.load(config.BASELINE_MODEL_PATHS["EGE-UNet"]),
        "PA-Net": torch.load(config.BASELINE_MODEL_PATHS["PA-Net"]),
        "AGCAF-Net": torch.load(config.BASELINE_MODEL_PATHS["AGCAF-Net"]),
        "PVTFormer": torch.load(config.BASELINE_MODEL_PATHS["PVTFormer"]),
        "Ours": model  # DGA-Net（基准）
    }

    # 4. 评估所有模型在各大小分层的DPC（表5核心指标）
    table_data = []
    for model_name, model in baseline_models.items():
        model = model.to(config.DEVICE)
        result = {"Method": model_name}

        # 评估每个大小分层的DPC
        for strata_name, strata_data in size_strata.items():
            strata_loader = DataLoader(
                strata_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2
            )
            # 多次重复评估（计算标准差和95%CI）
            dpc_vals = []
            for _ in range(args.num_repeats):
                metrics = evaluate_model(model, strata_loader, config.DEVICE, class_id=2)
                dpc_vals.append(metrics["DPC"]["mean"])

            # 格式化结果（匹配论文表5格式）
            mean_dpc = np.mean(dpc_vals)
            std_dpc = np.std(dpc_vals, ddof=1)
            ci_interval = stats.t.interval(0.95, len(dpc_vals) - 1, loc=mean_dpc, scale=stats.sem(dpc_vals))
            ci_lower, ci_upper = ci_interval
            result[strata_name] = f"{mean_dpc:.4f}({std_dpc:.4f})\n[{ci_lower:.4f},{ci_upper:.4f}]"

        table_data.append(result)

    # 5. 保存表5结果
    df = pd.DataFrame(table_data)
    save_path = os.path.join(args.save_dir, "Table5_Tumor_Size_Influence.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    logger.log({"event": "table5_reproduce_end", "info": f"表5复现完成，保存路径：{save_path}"})
    logger.close()
    print("\n表5（肿瘤大小影响）复现结果预览：")
    print(df.to_string(index=False))
    return save_path


def reproduce_table6(args):
    """复现表6（高斯噪声影响），论文4.4.3节：在输入图像添加高斯噪声后评估"""
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.model_save_path, exist_ok=True)
    logger = Logger("reproduce_table6")
    logger.log({"event": "table6_reproduce_start", "info": "开始复现表6（高斯噪声影响）"})

    # 1. 获取所有消融变体（同表3/4，评估噪声鲁棒性）
    ablation_variants = get_ablation_variants()
    # 2. 加载添加高斯噪声的LiTS2017测试集（论文4.4.3节：模拟设备电子噪声）
    test_dataset = get_dataset(
        config.PROCESSED_DATA_PATH, "LiTS2017", split="test", add_gaussian_noise=True, noise_std=0.1
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 3. 评估所有消融变体在噪声数据上的性能（肿瘤分割任务，class_id=2）
    table_data = []
    for model_name, model in ablation_variants.items():
        # 加载预训练的消融模型（复用表4训练好的权重，避免重复训练）
        model_path = os.path.join(args.model_save_path, f"{model_name}_tumor_best.pth")
        if not os.path.exists(model_path):
            # 若未训练，先训练该消融变体
            model = train_ablation_model(model, model_name, "tumor", args)
        else:
            model.load_state_dict(torch.load(model_path))
            model = model.to(config.DEVICE)

        # 多次重复评估（计算标准差和95%CI）
        metrics_list = []
        for _ in range(args.num_repeats):
            metrics = evaluate_model(model, test_loader, config.DEVICE, class_id=2)
            metrics_list.append(metrics)

        # 整理结果（匹配论文表6格式）
        result = {"Method": model_name}
        for metric in ["DPC", "DG", "VOE", "RAVD", "ASSD"]:
            metric_vals = [m[metric]["mean"] for m in metrics_list]
            mean_val = np.mean(metric_vals)
            std_val = np.std(metric_vals, ddof=1)
            ci_interval = stats.t.interval(
                0.95, len(metric_vals) - 1, loc=mean_val, scale=stats.sem(metric_vals)
            )
            ci_lower, ci_upper = ci_interval

            if metric in ["DPC", "DG", "VOE", "RAVD"]:
                result[metric] = f"{mean_val:.4f}({std_val:.4f})\n[{ci_lower:.4f},{ci_upper:.4f}]"
            else:
                result[metric] = f"{mean_val:.2f}({std_val:.2f})\n[{ci_lower:.2f},{ci_upper:.2f}]"

        table_data.append(result)

    # 4. 保存表6结果
    df = pd.DataFrame(table_data)
    save_path = os.path.join(args.save_dir, "Table6_Gaussian_Noise_Influence.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    logger.log({"event": "table6_reproduce_end", "info": f"表6复现完成，保存路径：{save_path}"})
    logger.close()
    print("\n表6（高斯噪声影响）复现结果预览：")
    print(df.to_string(index=False))
    return save_path


def main():
    args = parse_args()

    # 固定随机种子（确保实验可复现，论文4.4节所有消融实验种子=42）
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # 根据目标表格执行对应复现逻辑
    if args.ablation_target == "table3":
        # 表3：肝分割模块消融（任务=liver）
        reproduce_table3_table4(args, task="liver", target_table="Table3")
    elif args.ablation_target == "table4":
        # 表4：肿瘤分割模块消融（任务=tumor）
        reproduce_table3_table4(args, task="tumor", target_table="Table4")
    elif args.ablation_target == "table5":
        # 表5：肿瘤大小影响
        reproduce_table5(args)
    elif args.ablation_target == "table6":
        # 表6：高斯噪声影响
        reproduce_table6(args)
    else:
        raise ValueError(f"无效消融目标表格：{args.ablation_target}，仅支持table3/table4/table5/table6")


if __name__ == "__main__":
    main()