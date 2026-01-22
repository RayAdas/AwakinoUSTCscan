import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.file_io import FileIO

if __name__ == "__main__":
    FileIO.init()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from rebuild.dataset import DeepImgDataset # import src for torch.load
    dataset = torch.load(os.path.join(FileIO.curr_rebuild_dataset_path, "dataset.pt"), weights_only=False)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    from rebuild.unet import UNet3D, build_gaussian_heatmap, heatmap_loss

    model = UNet3D(in_ch=1, base_ch=32).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    # -------------------------
    # 训练超参数
    # -------------------------
    num_epochs = 100
    sigma = 2.0        # heatmap 宽度（t 方向，单位：bin）
    log_interval = 20

    # -------------------------
    # 训练循环
    # -------------------------
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for i, batch in enumerate(pbar):
            x = batch["input"].to(device)    # (B,H,W,T)
            r = batch["target"].to(device)   # (B,H,W)

            # shape 对齐
            x = x.unsqueeze(1)               # (B,1,H,W,T)

            # 构造 heatmap supervision
            s_gt = build_gaussian_heatmap(
                r,
                wave_len=x.shape[-1],
                sigma=sigma
            )                                # (B,H,W,T)
            s_gt = s_gt.unsqueeze(1)         # (B,1,H,W,T)

            # forward
            pred = model(x)

            loss = heatmap_loss(pred, s_gt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % log_interval == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4e}"
                )

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch+1}] avg loss = {avg_loss:.6e}")

        # -------------------------
        # checkpoint（可选）
        # -------------------------
        if (epoch + 1) % 10 == 0:
            ckpt_path = FileIO.curr_rebuild_ckpt_dir / f"unet3d_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path
            )
            print(f"Saved checkpoint: {ckpt_path}")

    print("Training finished.")