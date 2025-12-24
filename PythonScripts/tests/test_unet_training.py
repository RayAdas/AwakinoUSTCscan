"""Test script for training U-Net on depth reconstruction task."""

import torch
from torch.utils.data import DataLoader, random_split

from rebuild.dataset import DeepImgDataset
from rebuild.strategies import UNetStrategy

import matplotlib.pyplot as plt


def train_unet():
    """Train U-Net model on synthetic depth data."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset parameters
    n_samples = 2000
    receptive_field_size = 41
    d_input = 128  # wave_len
    
    print(f"\n{'='*50}")
    print("Creating dataset...")
    print(f"{'='*50}")
    
    # Create dataset
    dataset = DeepImgDataset(
        receptive_field_size=receptive_field_size,
        n_samples=n_samples,
        d_input=d_input
    )
    
    print(f"Total samples: {len(dataset)}")
    print(f"Input shape: {dataset[0]['input'].shape}")  # (H, W, wave_len)
    print(f"Target shape: {dataset[0]['target'].shape}")  # (H, W)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nSplit: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n{'='*50}")
    print("Building U-Net model...")
    print(f"{'='*50}")
    
    # Create strategy
    strategy = UNetStrategy(
        input_channels=d_input,
        spatial_size=receptive_field_size,
        base_channels=64,
        bilinear=True,
        dropout=0.1,
        device=device
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in strategy.model.parameters())
    trainable_params = sum(p.numel() for p in strategy.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print(f"\n{'='*50}")
    print("Training...")
    print(f"{'='*50}")
    
    # Train model
    history = strategy.train(
        train_loader,
        val_loader,
        epochs=50,
        lr=1e-3,
        weight_decay=1e-5,
        patience=10,
        save_path="models/rebuild_unet_best.pt"
    )
    
    print(f"\n{'='*50}")
    print("Evaluating on test set...")
    print(f"{'='*50}")
    
    # Evaluate
    metrics = strategy.evaluate(test_loader)
    
    print("\nTest Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Visualize results
    print(f"\n{'='*50}")
    print("Visualizing results...")
    print(f"{'='*50}")
    
    visualize_results(strategy, test_dataset, device, history)
    
    print("\nTraining complete!")


def visualize_results(strategy, test_dataset, device, history):
    """Visualize training history and sample predictions."""
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    
    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training History")
    axes[0].legend()
    axes[0].grid(True)
    
    # Show sample predictions
    sample_idx = 0
    sample = test_dataset[sample_idx]
    
    wave_input = sample["input"].unsqueeze(0).to(device)  # (1, H, W, wave_len)
    depth_target = sample["target"].cpu().numpy()
    
    with torch.no_grad():
        depth_pred = strategy.predict(wave_input).squeeze().cpu().numpy()
    
    # Plot depth maps
    vmin = min(depth_target.min(), depth_pred.min())
    vmax = max(depth_target.max(), depth_pred.max())
    
    im1 = axes[1].imshow(depth_target, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Ground Truth (Max: {depth_target.max():.4f})")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Show more predictions
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(3):
        sample = test_dataset[i]
        wave_input = sample["input"].unsqueeze(0).to(device)
        depth_target = sample["target"].cpu().numpy()
        
        with torch.no_grad():
            depth_pred = strategy.predict(wave_input).squeeze().cpu().numpy()
        
        vmin = min(depth_target.min(), depth_pred.min())
        vmax = max(depth_target.max(), depth_pred.max())
        
        # Target
        im_target = axes2[0, i].imshow(depth_target, cmap='viridis', vmin=vmin, vmax=vmax)
        axes2[0, i].set_title(f"Sample {i+1}: Ground Truth")
        axes2[0, i].axis('off')
        plt.colorbar(im_target, ax=axes2[0, i], fraction=0.046)
        
        # Prediction
        im_pred = axes2[1, i].imshow(depth_pred, cmap='viridis', vmin=vmin, vmax=vmax)
        axes2[1, i].set_title(f"Sample {i+1}: Prediction")
        axes2[1, i].axis('off')
        plt.colorbar(im_pred, ax=axes2[1, i], fraction=0.046)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_unet()
