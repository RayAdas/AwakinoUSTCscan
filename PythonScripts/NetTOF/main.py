"""Command line entry point for training NetTOF strategies."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from .dataset import SyntheticEchoDataset
from .strategies import STRATEGY_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NetTOF strategies on synthetic data.")
    parser.add_argument("--strategy", type=str, default="regression", help="Strategy key to use.")
    parser.add_argument("--n_iter_outer", type=int, default=2, help="Number of echoes per waveform.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for loaders.")
    parser.add_argument("--total_samples", type=int, default=4096, help="Total synthetic samples to generate.")
    parser.add_argument("--waveform_length", type=int, default=2048, help="Number of time points per waveform.")
    parser.add_argument("--time_span", type=float, default=6.0e-5, help="Total time span of each waveform in seconds.")
    parser.add_argument("--noise_std", type=float, default=0.015, help="Gaussian noise standard deviation.")
    parser.add_argument("--noise_bias_std", type=float, default=0.0025, help="Background bias noise level.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split proportion.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split proportion.")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1.0e-5, help="Weight decay factor.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("models"),
        help="Directory to store trained models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SyntheticEchoDataset(
        n_samples=args.total_samples,
        n_iter_outer=args.n_iter_outer,
        waveform_length=args.waveform_length,
        time_span=args.time_span,
        noise_std=args.noise_std,
        noise_bias_std=args.noise_bias_std,
        seed=args.seed,
    )

    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    strategy_cls = STRATEGY_REGISTRY.get(args.strategy)
    if strategy_cls is None:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy '{args.strategy}'. Available: {available}")

    strategy = strategy_cls(n_iter_outer=args.n_iter_outer, waveform_length=args.waveform_length, device=device)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.save_dir / f"{args.strategy}_best.pt"

    print("Starting training...")
    strategy.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        denormalize_fn=dataset.denormalize_params,
        save_path=save_path,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    print("Evaluating on test set...")
    metrics = strategy.evaluate(test_loader, denormalize_fn=dataset.denormalize_params)

    print("Test MAE:")
    for name, value in metrics["mae"].items():
        print(f"  {name}: {value:.6e}")

    print("Test RMSE:")
    for name, value in metrics["rmse"].items():
        print(f"  {name}: {value:.6e}")

    from EchoModel import echo_function, echo_info_default
    import numpy as np
    t = np.linspace(0, args.time_span, args.waveform_length)
    example_waveform = echo_function(t, **echo_info_default._asdict())
    example_waveform = torch.tensor(example_waveform, dtype=torch.float32).unsqueeze(0).to(device)
    predicted_params = strategy.predict(example_waveform, denormalize_fn=dataset.denormalize_params)
    print("Example prediction for default echo parameters:")
    for name, value in zip(echo_info_default._fields, predicted_params.squeeze().cpu().numpy()):
        print(f"  {name}: {value:.6e}")


if __name__ == "__main__":
    main()
