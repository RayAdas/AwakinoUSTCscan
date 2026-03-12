import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.file_io import FileIO
from rebuild.dataset import DeepImgDataset
from rebuild.unet import UNet3D


# ======================
# Hyperparameters
# ======================
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

VAL_RATIO = 0.1
EARLY_STOP_PATIENCE = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TGT_KERNEL = torch.ones((1, 1))


def heatmap_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    center_weight = 5.0
    weight_map = torch.ones_like(target)  # (B, 1, H, W, T)
    _, _, h, w, _ = target.shape

    c_h, c_w = h // 2, w // 2
    weight_map[:, :, c_h - 2:c_h + 3, c_w - 2:c_w + 3, :] = center_weight

    loss = F.mse_loss(pred, target, reduction="none")
    loss = loss * weight_map
    return loss.mean()


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        x: torch.Tensor = batch["input"].to(DEVICE)   # (B, H, W, T)
        r: torch.Tensor = batch["target"].to(DEVICE)  # (B, H, W)

        x = x.unsqueeze(1)  # (B, 1, H, W, T)

        s_gt = DeepImgDataset.defects_to_waves(
            r,
            TGT_KERNEL,
            receptive_field_size=r.shape[1],
            sigma=DeepImgDataset.SIGMA,
        ).unsqueeze(1)  # (B, 1, H, W, T)

        optimizer.zero_grad()
        pred = model(x)
        loss = heatmap_loss(pred, s_gt)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Val", leave=False):
        x: torch.Tensor = batch["input"].to(DEVICE)
        r: torch.Tensor = batch["target"].to(DEVICE)

        x = x.unsqueeze(1)

        s_gt = DeepImgDataset.defects_to_waves(
            r,
            TGT_KERNEL,
            receptive_field_size=r.shape[1],
            sigma=DeepImgDataset.SIGMA,
        ).unsqueeze(1)

        pred = model(x)
        loss = heatmap_loss(pred, s_gt)

        total_loss += loss.item()

    return total_loss / len(loader)


def load_pretrained_model() -> UNet3D:
    model = UNet3D(in_ch=1).to(DEVICE)
    ckpt_path = FileIO.rebuild_model_path
    if ckpt_path is None:
        raise ValueError("No CurrentRebuildModel selected in config.ini")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)

    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state, strict=False)

    print(f"Loaded pretrained model from: {ckpt_path}")
    return model


if __name__ == "__main__":
    FileIO.init()

    if FileIO.curr_rebuild_dataset_path is None:
        raise ValueError("No CurrentRebuildDataset selected in config.ini")

    dataset_path = os.path.join(FileIO.curr_rebuild_dataset_path, "dataset.pt")
    dataset = torch.load(dataset_path, weights_only=False)

    print("using device:", DEVICE)
    print(f"dataset: {dataset_path}")
    print(f"dataset size: {len(dataset)}")

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = load_pretrained_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    save_path = os.path.join(FileIO.models_path, f"unet3d_fine_best_{timestamp}.pth")
    os.makedirs(FileIO.models_path, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{NUM_EPOCHS}]")

        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val   Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "finetune_from": FileIO.rebuild_model_path,
                    "dataset": FileIO.curr_rebuild_dataset_path,
                },
                save_path,
            )
            print(f"[OK] Best fine-tuned model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Fine-tuning finished.")
