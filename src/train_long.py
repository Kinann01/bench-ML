#!/usr/bin/env python3

import numpy as np
import argparse
from pathlib import Path

from encoder import TS2Vec
from pipeline_config import PipelineConfig

from constants import ROOT

def main():
    parser = argparse.ArgumentParser(description="TS2Vec training")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to preprocessed .npy training data')
    parser.add_argument('--output', type=str, default='models/ts2vec.pt')
    parser.add_argument('--conf', type=str, default=None,
                        help='Path to pipeline.conf (uses defaults if omitted)')
    parser.add_argument('--plot-dir', type=str, default='plots_training')
    parser.add_argument('--plot-every', type=int, default=5)
    args = parser.parse_args()

    pcfg = PipelineConfig(args.conf)
    tcfg = pcfg.training

    # Load data
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    padded_runs = np.load(str(data_path))
    print(f"Data loaded: shape={padded_runs.shape}")

    # Create model
    model = TS2Vec(
        input_dim=padded_runs.shape[2],
        hidden_dim=tcfg["hidden_dim"],
        repr_dim=tcfg["repr_dim"],
        depth=tcfg["depth"],
        lr=tcfg["learning_rate"],
        mask_ratio=tcfg["mask_ratio"],
        weight_decay=tcfg["weight_decay"],
        eta_min=tcfg["eta_min"],
    )

    # Train
    try:
        model.fit(
            padded_runs,
            epochs=tcfg["epochs"],
            batch_size=tcfg["batch_size"],
            plot_every=args.plot_every,
            plot_dir=args.plot_dir,
        )
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(model.history['train_loss'])} epochs")

    # Save
    model_path = Path(args.output)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    n_epochs = len(model.history['train_loss'])
    print(f"\nTraining complete:")
    print(f"  Epochs: {n_epochs}/{tcfg['epochs']}")
    if n_epochs > 0:
        print(f"  Final loss: {model.history['train_loss'][-1]:.6f}")
    print(f"  Model: {model_path}")


if __name__ == "__main__":
    main()
