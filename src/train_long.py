"""
TS2Vec training script for benchmark anomaly detection.

Usage:
    python3 train_long.py --epochs 15 --data training_data_all_years.npy
    python3 train_long.py --epochs 50 --repr-dim 320 --hidden-dim 128 --depth 10
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import argparse
from scipy.spatial.distance import pdist, cdist

from encoder import TS2Vec, PAD_VALUE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def compute_real_lengths(X: np.ndarray) -> np.ndarray:
    """Compute real (non-padded) lengths for each sample."""
    if X.ndim == 3:
        x_flat = X.squeeze(-1)
    else:
        x_flat = X
    lengths = []
    for i in range(len(x_flat)):
        nz = np.where(x_flat[i] != PAD_VALUE)[0]
        lengths.append(nz[-1] + 1 if len(nz) > 0 else 0)
    return np.array(lengths)


def plot_tsne(embeddings: np.ndarray, lengths: np.ndarray, plot_dir: Path):
    """
    t-SNE projection of embeddings, colored by sequence length.
    If similar benchmarks cluster together, the encoder is learning 
    meaningful representations.
    """
    from sklearn.manifold import TSNE

    logger.info("Running t-SNE on embeddings (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Color by sequence length
    sc1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=lengths,
                          cmap='viridis', s=8, alpha=0.6)
    axes[0].set_title('t-SNE — Colored by Sequence Length')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    plt.colorbar(sc1, ax=axes[0], label='Length')

    # Color by mean value (proxy for "what config this might be")
    # Compute mean of non-padded values
    means = []
    for i in range(len(embeddings)):
        means.append(embeddings[i].mean())
    means = np.array(means)

    sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=means,
                          cmap='coolwarm', s=8, alpha=0.6)
    axes[1].set_title('t-SNE — Colored by Mean Embedding Value')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    plt.colorbar(sc2, ax=axes[1], label='Mean Embedding')

    plt.suptitle('TS2Vec Embedding Space Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(plot_dir / "tsne_embeddings.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved t-SNE plot: {plot_dir}/tsne_embeddings.png")


def plot_nearest_neighbors(X: np.ndarray, embeddings: np.ndarray,
                           lengths: np.ndarray, plot_dir: Path, n_queries: int = 5, k: int = 5):
    """
    For n_queries random samples, find k nearest neighbors in embedding space
    and plot them side by side. If the encoder works, neighbors should look similar.
    """
    logger.info(f"Finding {k} nearest neighbors for {n_queries} query samples...")

    # Compute all pairwise distances
    dists = cdist(embeddings, embeddings, metric='cosine')

    # Pick random queries
    rng = np.random.RandomState(42)
    query_indices = rng.choice(len(embeddings), size=n_queries, replace=False)

    for q_idx, query_i in enumerate(query_indices):
        # Get k+1 nearest (first is itself)
        neighbor_indices = np.argsort(dists[query_i])[:k + 1]
        neighbor_indices = neighbor_indices[neighbor_indices != query_i][:k]

        all_indices = [query_i] + list(neighbor_indices)
        n_plots = len(all_indices)

        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots))
        if n_plots == 1:
            axes = [axes]

        for j, (ax, idx) in enumerate(zip(axes, all_indices)):
            series = X[idx].flatten()
            real_len = int(lengths[idx])
            series_real = series[:real_len]

            cosine_dist = dists[query_i, idx] if idx != query_i else 0.0
            label = "QUERY" if j == 0 else f"Neighbor {j} (dist={cosine_dist:.4f})"
            color = '#E53935' if j == 0 else '#1E88E5'

            ax.plot(series_real, linewidth=1, color=color, alpha=0.8)
            ax.set_title(f"{label} — Length: {real_len}", fontsize=10)
            ax.set_ylabel("Value")

        axes[-1].set_xlabel("Timestep")
        fig.suptitle(f"Query #{q_idx + 1} — Nearest Neighbors in Embedding Space",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(plot_dir / f"neighbors_query_{q_idx + 1}.png"), dpi=120)
        plt.close(fig)

    logger.info(f"Saved {n_queries} nearest-neighbor plots to {plot_dir}/")


def main():
    parser = argparse.ArgumentParser(description="TS2Vec Training for Benchmark Anomaly Detection")
    parser.add_argument('--epochs', type=int, default=15,
                        help='Training epochs (default: 15)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--repr-dim', type=int, default=320,
                        help='Representation dimension (default: 320)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Encoder hidden dimension (default: 128)')
    parser.add_argument('--depth', type=int, default=10,
                        help='Number of dilated conv blocks (default: 10)')
    parser.add_argument('--plot-every', type=int, default=5,
                        help='Save loss curve every N epochs (default: 5)')
    parser.add_argument('--data', type=str, default='data/training_data_long.npy',
                        help='Path to preprocessed .npy data file')
    parser.add_argument('--output', type=str, default='TS2Vec/ts2vec_long.pt',
                        help='Output path for trained model')
    parser.add_argument('--plot-dir', type=str, default='TS2Vec/plots_long',
                        help='Output directory for plots')
    parser.add_argument('--n-queries', type=int, default=10,
                        help='Number of nearest-neighbor query plots (default: 10)')
    parser.add_argument('--tsne-samples', type=int, default=3000,
                        help='Number of samples for t-SNE (default: 3000)')
    parser.add_argument('--skip-diagnostics', action='store_true',
                        help='Skip post-training diagnostics (t-SNE, neighbors)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TS2Vec Training — Benchmark Anomaly Detection")
    logger.info("=" * 60)

    # Step 1: Load data
    data_path = ROOT / args.data
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    X = np.load(str(data_path))
    logger.info(f"Data loaded: shape={X.shape}")

    # Pre-compute lengths for diagnostics later
    lengths = compute_real_lengths(X)
    logger.info(f"Sequence lengths — min: {lengths.min()}, max: {lengths.max()}, "
                f"median: {np.median(lengths):.0f}, mean: {lengths.mean():.0f}")

    # Step 2: Create model
    model = TS2Vec(
        input_dim=X.shape[2],
        hidden_dim=args.hidden_dim,
        repr_dim=args.repr_dim,
        depth=args.depth,
        lr=args.lr,
    )

    # Step 3: Train (Ctrl+C will gracefully save + run diagnostics)
    interrupted = False
    logger.info("Starting training...")
    try:
        history = model.fit(
            X,
            epochs=args.epochs,
            batch_size=args.batch_size,
            plot_every=args.plot_every,
            plot_dir=args.plot_dir,
        )
    except KeyboardInterrupt:
        interrupted = True
        history = model.history
        print(f"\n\n⚠️  Interrupted! Saving model after {len(history['train_loss'])} epochs...")

    # Step 4: Save model
    model_path = ROOT / args.output
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    plot_dir = ROOT / args.plot_dir
    plot_dir.mkdir(exist_ok=True)

    if not args.skip_diagnostics:
        # Step 5: Encode samples for diagnostics
        n_tsne = min(args.tsne_samples, len(X))
        logger.info(f"Encoding {n_tsne} samples for diagnostics...")

        # Use a random subset for t-SNE (full dataset is too slow)
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(X), size=n_tsne, replace=False)
        X_sample = X[sample_idx]
        lengths_sample = lengths[sample_idx]

        embeddings = model.encode(X_sample)
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Embedding stats — mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}, "
                    f"norm_mean: {np.linalg.norm(embeddings, axis=1).mean():.4f}")

        # Step 6: t-SNE visualization
        plot_tsne(embeddings, lengths_sample, plot_dir)

        # Step 7: Nearest-neighbor plots
        plot_nearest_neighbors(X_sample, embeddings, lengths_sample, plot_dir,
                              n_queries=args.n_queries, k=5)

        # Step 8: Embedding stats
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        norms = np.linalg.norm(embeddings, axis=1)
        axes[0].hist(norms, bins=30, edgecolor='black', color='#2196F3')
        axes[0].set_xlabel('Embedding L2 Norm')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Embedding Norm Distribution ({n_tsne} samples)')

        cos_dists = pdist(embeddings[:500], metric='cosine')  # Subset for speed
        axes[1].hist(cos_dists, bins=50, edgecolor='black', color='#FF9800')
        axes[1].set_xlabel('Cosine Distance')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Pairwise Cosine Distance (500 samples)')
        plt.tight_layout()
        plt.savefig(str(plot_dir / "embedding_stats.png"), dpi=150)
        plt.close(fig)

    # Summary
    n_completed = len(history['train_loss'])
    status = "Interrupted" if interrupted else "Complete"
    print(f"\n{'='*60}")
    print(f"TS2Vec Training {status}!")
    print(f"  Model: hidden={args.hidden_dim}, repr_dim={args.repr_dim}, depth={args.depth}")
    print(f"  Samples: {len(X)}, Epochs: {n_completed}/{args.epochs}")
    if n_completed > 0:
        print(f"  Final loss: {history['train_loss'][-1]:.6f}")
    print(f"  Model saved to: {args.output}")
    print(f"  Plots saved to: {args.plot_dir}/")
    if not args.skip_diagnostics:
        print(f"  Diagnostics:")
        print(f"    - tsne_embeddings.png  (are clusters forming?)")
        print(f"    - neighbors_query_*.png  (do neighbors look similar?)")
        print(f"    - embedding_stats.png  (distance distribution)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
