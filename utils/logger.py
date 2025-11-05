import os
import json
from datetime import datetime
from typing import Dict, Any


class Logger:
    """
    日志记录工具类，用于记录S2DA-Net训练/评估过程中的关键信息
    支持实时日志打印、JSON格式结果保存，适配论文4.1-4.3节实验结果的记录需求
    """

    def __init__(self, log_dir: str = "./logs"):
        """
        初始化日志器，创建日志目录和日志文件

        参数：
        - log_dir: 日志存储目录，默认./logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 生成日志文件名（含时间戳，避免重复）
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"s2da_net_log_{time_str}.json")
        self.logs = []  # 存储所有日志记录的列表

        # 记录初始化信息（包含论文对应实验配置）
        init_log = {
            "timestamp": time_str,
            "event": "logger_initialized",
            "info": "S2DA-Net实验日志记录开始，基于1-s2.0-S0010482524004840-main.pdf实验设计"
        }
        self.log(init_log)
        print(f"日志器初始化完成，日志文件保存路径：{self.log_file}")

    def log(self, record: Dict[str, Any]):
        """
        记录一条日志，自动添加时间戳并保存到文件

        参数：
        - record: 日志字典，需包含实验关键信息（如epoch、loss、metrics等）
        """
        # 补充当前时间戳
        record["current_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(record)

        # 实时打印日志（关键信息）
        if "epoch" in record:
            print(f"[{record['current_time']}] Epoch {record['epoch']:04d} | "
                  f"Train Loss: {record.get('train_loss', 0.0):.4f} | "
                  f"Val Loss: {record.get('val_loss', 0.0):.4f} | "
                  f"Val DG: {record.get('DG', 0.0):.4f}")
        elif "event" in record:
            print(f"[{record['current_time']}] Event: {record['event']} | Info: {record['info']}")

        # 保存到JSON文件（追加模式，确保断电不丢失）
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

    def save_final_results(self, final_metrics: Dict[str, Any], dataset_name: str):
        """
        保存最终实验结果，适配论文4.1节（LiTS2017）和4.3节（3DIRCADb）的结果记录需求

        参数：
        - final_metrics: 最终评估指标字典（如DPC、DG、VOE等）
        - dataset_name: 数据集名称（如"LiTS2017"、"3DIRCADb"）
        """
        final_log = {
            "event": "experiment_completed",
            "dataset": dataset_name,
            "final_metrics": final_metrics,
            "info": f"{dataset_name}数据集实验完成，结果符合1-s2.0-S0010482524004840-main.pdf评估标准"
        }
        self.log(final_log)

        # 额外生成纯文本结果文件（方便直接引用到论文表格）
        result_txt = os.path.join(self.log_dir, f"final_results_{dataset_name}.txt")
        with open(result_txt, "w", encoding="utf-8") as f:
            f.write(f"S2DA-Net {dataset_name} 实验最终结果\n")
            f.write(f"评估时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            for metric, value in final_metrics.items():
                if metric in ["DPC", "DG"]:
                    f.write(f"{metric}: {value * 100:.2f}%\n")  # 百分比格式（论文表格风格）
                else:
                    f.write(f"{metric}: {value:.4f}\n")
        print(f"最终结果已保存到：{result_txt}")

    def close(self):
        """关闭日志器，记录实验结束信息"""
        close_log = {
            "event": "logger_closed",
            "info": "S2DA-Net实验日志记录结束，所有结果已保存"
        }
        self.log(close_log)