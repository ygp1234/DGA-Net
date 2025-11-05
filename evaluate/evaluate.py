import numpy as np
from utils.metrics import dice_coefficient, volume_overlap_error, relative_volume_difference, \
    average_symmetric_surface_distance


def compute_metrics(preds_list, labels_list):
    """计算论文中所有评估指标"""
    # 展平所有预测和标签
    all_preds = np.concatenate([p.flatten() for p in preds_list])
    all_labels = np.concatenate([l.flatten() for l in labels_list])

    # 计算全局Dice（DG）
    dg = dice_coefficient(all_preds, all_labels, class_id=2)  # 肿瘤类别为2

    # 计算单病例平均Dice（DPC）
    dpc_list = []
    for preds, labels in zip(preds_list, labels_list):
        for p, l in zip(preds, labels):
            dpc_list.append(dice_coefficient(p, l, class_id=2))
    dpc = np.mean(dpc_list)

    # 计算其他指标
    voe = volume_overlap_error(all_preds, all_labels, class_id=2)
    ravd = relative_volume_difference(all_preds, all_labels, class_id=2)
    assd = average_symmetric_surface_distance(all_preds, all_labels, class_id=2)

    return {
        "DPC": dpc,
        "DG": dg,
        "VOE": voe,
        "RAVD": ravd,
        "ASSD": assd
    }


def evaluate_model(model_path, dataloader, device):
    """加载模型并评估"""
    import torch
    from models.dag_net import DAGNet
    from config.config import config

    model = DAGNet(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            imgs, labels = batch
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    metrics = compute_metrics(all_preds, all_labels)
    return metrics


if __name__ == "__main__":
    # 示例：评估测试集
    import torch
    from data.datasets import LiTSDataset
    from config.config import config

    test_dataset = LiTSDataset(config.PROCESSED_DATA_PATH, split="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    metrics = evaluate_model("./checkpoints/best_s2da_net.pth", test_loader, config.DEVICE)
    print("Test Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")