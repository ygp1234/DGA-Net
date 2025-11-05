import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config.config import config
from data.datasets import LiTSDataset
from models.dag_net import DAGNet
from train.loss import MixedLoss
from evaluate.evaluate import compute_metrics
from utils.logger import Logger


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)

        # 前向传播
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def val_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            # 保存预测结果和标签（用于后续指标计算）
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    # 计算评估指标
    metrics = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics


def train():
    # 初始化日志
    logger = Logger("./logs")

    # 加载数据集
    train_dataset = LiTSDataset(config.PROCESSED_DATA_PATH, split="train")
    val_dataset = LiTSDataset(config.PROCESSED_DATA_PATH, split="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 初始化模型、损失函数、优化器
    model = DAGNet(config).to(config.DEVICE)
    criterion = MixedLoss(alpha=0.5)  # CE+Dice混合损失
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        momentum=0.9
    )
    # 多项式学习率衰减
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (1 - epoch / config.MAX_EPOCHS) ** config.POLY_POWER
    )

    # 训练循环
    best_dice = 0.0
    for epoch in range(config.MAX_EPOCHS):
        print(f"Epoch {epoch + 1}/{config.MAX_EPOCHS}")

        # 训练和验证
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_metrics = val_epoch(model, val_loader, criterion, config.DEVICE)

        # 更新学习率
        lr_scheduler.step()

        # 日志记录
        logger.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics
        })

        # 保存最佳模型
        if val_metrics["DG"] > best_dice:
            best_dice = val_metrics["DG"]
            torch.save(model.state_dict(), "./checkpoints/best_s2da_net.pth")
            print(f"Best model saved with DG: {best_dice:.4f}")

        # 打印结果
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Metrics: DPC={val_metrics['DPC']:.4f}, DG={val_metrics['DG']:.4f}, VOE={val_metrics['VOE']:.4f}")

    logger.close()


if __name__ == "__main__":
    train()