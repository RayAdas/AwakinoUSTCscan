import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm

from utils.file_io import FileIO
from rebuild.dataset import DeepImgDataset
from rebuild.unet import UNet3D

# ======================
# 超参数
# ======================
BATCH_SIZE = 4
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

VAL_RATIO = 0.1
EARLY_STOP_PATIENCE = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TGT_KERNEL = torch.ones((1,1))

def heatmap_loss(pred, target):
    # 简单的加权 MSE
    # 给予 target > 0 的区域更高的权重 (例如 100 倍)
    weight = torch.ones_like(target)
    weight[target > 0.01] = 100.0  # 前景权重
    
    loss = F.mse_loss(pred, target, reduction='none')
    loss = loss * weight
    return loss.mean()

# ======================
# 训练与验证函数
# ======================
def train_one_epoch(model:torch.nn.Module, loader, optimizer:torch.optim.Optimizer):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        x:torch.Tensor = batch["input"].to(DEVICE)      # (B,H,W,T)
        r:torch.Tensor = batch["target"].to(DEVICE)     # (B,H,W)

        x = x.unsqueeze(1)                 # (B,1,H,W,T)

        s_gt = DeepImgDataset.defects_to_waves(
            r,
            TGT_KERNEL,
            receptive_field_size=r.shape[1],
            sigma=DeepImgDataset.SIGMA,
        ).unsqueeze(1)                     # (B,1,H,W,T)

        optimizer.zero_grad()
        pred = model(x)
        loss = heatmap_loss(pred, s_gt)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Val", leave=False):
        x = batch["input"].to(DEVICE)
        r = batch["target"].to(DEVICE)

        x = x.unsqueeze(1)

        s_gt = DeepImgDataset.defects_to_waves(
            r,
            TGT_KERNEL,
            receptive_field_size=r.shape[1],
            sigma=DeepImgDataset.SIGMA,
        ).unsqueeze(1)                     # (B,1,H,W,T)

        pred = model(x)
        loss = heatmap_loss(pred, s_gt)

        total_loss += loss.item()

    return total_loss / len(loader)


# ======================
# 主入口
# ======================
if __name__ == "__main__":
    FileIO.init()

    print("using device: ", DEVICE)

    dataset = torch.load(os.path.join(FileIO.curr_rebuild_dataset_path, "dataset.pt"), weights_only=False)

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ------------------
    # 模型 & 优化器
    # ------------------
    model = UNet3D(in_ch=1).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ------------------
    # Early stopping & best model
    # ------------------
    best_val_loss = float("inf")
    epochs_no_improve = 0

    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    save_path = os.path.join(FileIO.models_path, f"unet3d_best_{timestamp}.pth")
    os.makedirs(FileIO.models_path, exist_ok=True)

    # ------------------
    # 训练循环
    # ------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{NUM_EPOCHS}]")

        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val   Loss: {val_loss:.6f}")

        # ---- best model ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                save_path
            )
            print(f"[✓] Best model saved to {save_path}")

        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")

        # ---- early stopping ----
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training finished.")
