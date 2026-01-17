"""Test script for Transformer strategy in depth reconstruction."""

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from rebuild.dataset import DeepImgDataset
from rebuild.strategies import STRATEGY_REGISTRY


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
    
    # Show sample predictions in 3D surface plots
    fig2 = plt.figure(figsize=(18, 12))
    
    for i in range(3):
        sample = test_dataset[i]
        wave_input = sample["input"].unsqueeze(0).to(device)
        depth_target = sample["target"].cpu().numpy()
        
        with torch.no_grad():
            depth_pred = strategy.predict(wave_input).squeeze().cpu().numpy()
        
        vmin = min(depth_target.min(), depth_pred.min())
        vmax = max(depth_target.max(), depth_pred.max())
        
        # Create meshgrid for 3D plotting
        h, w = depth_target.shape
        x = np.arange(0, w, 1)
        y = np.arange(0, h, 1)
        X, Y = np.meshgrid(x, y)
        
        # Target - 3D surface
        ax_target = fig2.add_subplot(2, 3, i+1, projection='3d')
        surf_target = ax_target.plot_surface(X, Y, depth_target, cmap='viridis', 
                                             vmin=vmin, vmax=vmax, 
                                             edgecolor='none', alpha=0.9,
                                             rstride=1, cstride=1)
        ax_target.set_title(f"Sample {i+1}: Ground Truth")
        ax_target.set_xlabel('X')
        ax_target.set_ylabel('Y')
        ax_target.set_zlabel('Depth')
        ax_target.view_init(elev=30, azim=45)
        fig2.colorbar(surf_target, ax=ax_target, shrink=0.5, aspect=5)
        
        # Prediction - 3D surface
        ax_pred = fig2.add_subplot(2, 3, i+4, projection='3d')
        surf_pred = ax_pred.plot_surface(X, Y, depth_pred, cmap='viridis', 
                                         vmin=vmin, vmax=vmax, 
                                         edgecolor='none', alpha=0.9,
                                         rstride=1, cstride=1)
        ax_pred.set_title(f"Sample {i+1}: Prediction")
        ax_pred.set_xlabel('X')
        ax_pred.set_ylabel('Y')
        ax_pred.set_zlabel('Depth')
        ax_pred.view_init(elev=30, azim=45)
        fig2.colorbar(surf_pred, ax=ax_pred, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()


def test_transformer_strategy():
    """Test the Transformer strategy with a small dataset."""
    print("=" * 60)
    print("Testing Transformer Strategy for Depth Reconstruction")
    print("=" * 60)
    
    # Create dataset
    print("\n1. Creating dataset...")
    dataset = DeepImgDataset(
        receptive_field_size=41,
        sampling_interval=1e-3,
        conv_radius=5e-3,
        n_samples=100,  # Small dataset for testing
        d_input=128,
        batch_size=32
    )
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Input shape: {dataset.input.shape}")
    print(f"   Target shape: {dataset.tgt.shape}")
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create strategy
    print("\n2. Creating Transformer strategy...")
    strategy = STRATEGY_REGISTRY["transformer"](
        input_channels=128,
        spatial_size=41,
        embed_dim=128,  # Smaller for testing
        num_heads=4,
        num_layers=3,
        mlp_ratio=4,
        dropout=0.1
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in strategy.model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    sample_batch = next(iter(train_loader))
    input_data = sample_batch["input"]
    target_data = sample_batch["target"]
    print(f"   Batch input shape: {input_data.shape}")
    print(f"   Batch target shape: {target_data.shape}")
    
    with torch.no_grad():
        strategy.model.eval()
        output = strategy.model(input_data.to(strategy.device))
        print(f"   Batch output shape: {output.shape}")
        
        # Check shape consistency
        assert output.shape == target_data.shape, \
            f"Output shape {output.shape} doesn't match target shape {target_data.shape}"
        print("   ✓ Forward pass successful!")
    
    # Train for a few epochs
    print("\n4. Training model...")
    history = strategy.train(
        dataloader_train=train_loader,
        dataloader_val=val_loader,
        epochs=50,
        lr=1e-4,
        weight_decay=1e-5,
        patience=5
    )
    
    print(f"\n   Training history:")
    for epoch, metrics in enumerate(history, 1):
        print(f"   Epoch {epoch}: train_loss={metrics['train_loss']:.6f}, "
              f"val_loss={metrics['val_loss']:.6f}")
    
    # Evaluate
    print("\n5. Evaluating model...")
    test_metrics = strategy.evaluate(test_loader)
    print(f"   Test metrics: {test_metrics}")
    
    # Test prediction
    print("\n6. Testing prediction...")
    sample = dataset[0]
    wave_input = sample["input"]
    target_depth = sample["target"]
    
    predicted_depth = strategy.predict(wave_input.unsqueeze(0).to(strategy.device))
    print(f"   Predicted depth shape: {predicted_depth.shape}")
    print(f"   Target depth shape: {target_depth.shape}")
    
    # Calculate error
    error = torch.abs(predicted_depth.squeeze().to(strategy.device) - target_depth.to(strategy.device))
    print(f"   Mean absolute error: {error.mean().item():.6f}")
    print(f"   Max absolute error: {error.max().item():.6f}")
    
    # Visualize results
    print("\n7. Visualizing results...")
    visualize_results(strategy, test_dataset, strategy.device, history)
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_transformer_strategy()
