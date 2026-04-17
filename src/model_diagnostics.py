#!/usr/bin/env python3

import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

from detector import SteadyStateDetector
from encoder import TS2Vec
from pipeline_config import PipelineConfig

from constants import (ROOT, PAD_VALUE,
                    find_csv, resolve_iter_col) 

_detector = SteadyStateDetector()

def load_index(path: Path) -> Dict[tuple, List[Tuple[Path, int]]]:
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    return {key: [(Path(p), v) for p, v in entries]
            for key, entries in raw.items()}

def collect_samples(index: dict, n_samples: int, min_length: int):

    all_series = []
    all_lengths = []
    all_benchmarks = []

    for key, entries in index.items():
        benchmark_name = key[0]

        for run_dir, _ in entries:
            csv_path = find_csv(run_dir)
            if csv_path is None:
                continue
            try:
                df = pd.read_csv(csv_path)
                iter_col = resolve_iter_col(df)
                if df.empty or iter_col is None:
                    continue

                cutoff = _detector.detect_cutoff_index(df)
                if cutoff == 0:
                    continue

                series = df[iter_col].iloc[cutoff:].values.astype(float)
                if len(series) < min_length:
                    continue

                all_series.append(series)
                all_lengths.append(len(series))
                all_benchmarks.append(benchmark_name)
            except Exception:
                continue

    print(f"Collected {len(all_series)} measurements (min_length={min_length})")

    # Sample
    rng = np.random.RandomState(42)
    n = min(n_samples, len(all_series))
    idx = rng.choice(len(all_series), size=n, replace=False)

    return ([all_series[i] for i in idx],
            np.array([all_lengths[i] for i in idx]),
            [all_benchmarks[i] for i in idx])


def preprocess_and_encode(model, series_list):

    max_len = max(len(s) for s in series_list)
    n = len(series_list)

    data = np.full((n, max_len), PAD_VALUE, dtype=np.float32)
    for i, s in enumerate(series_list):
        s_log = np.log(s)
        median = np.median(s_log)
        q25, q75 = np.percentile(s_log, [25, 75])
        iqr = max(q75 - q25, 1e-10)
        s_norm = (s_log - median) / iqr
        data[i, :len(s_norm)] = s_norm

    data = np.expand_dims(data, axis=-1)
    print(f"Preprocessed: shape {data.shape}")

    embeddings = model.encode(data)
    print(f"Encoded: shape {embeddings.shape}")

    return data, embeddings


def plot_tsne_length(coords, lengths, output_path):

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=lengths,
                    cmap='viridis', s=10, alpha=0.6)
    ax.set_title('t-SNE — Colored by Sequence Length',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(sc, ax=ax, label='Length')
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_tsne_benchmark(coords, benchmarks, output_path):

    unique = sorted(set(benchmarks))
    bench_to_id = {name: i for i, name in enumerate(unique)}
    ids = np.array([bench_to_id[b] for b in benchmarks])

    cmap = plt.colormaps.get_cmap('tab20').resampled(len(unique))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c=ids,
               cmap=cmap, s=10, alpha=0.6)
    ax.set_title('t-SNE — Colored by Benchmark Name',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    if len(unique) <= 25:
        handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=cmap(bench_to_id[b]),
                          markersize=6, label=b)
                   for b in unique]
        ax.legend(handles=handles, fontsize=7, loc='best',
                  ncol=2 if len(unique) > 12 else 1)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_neighbors(data, embeddings, lengths, output_dir, n_queries=10, k=5):

    dists = cdist(embeddings, embeddings, metric='cosine')
    rng = np.random.RandomState(42)
    queries = rng.choice(len(embeddings), size=n_queries, replace=False)

    for q_idx, query_i in enumerate(queries):
        neighbors = np.argsort(dists[query_i])
        neighbors = neighbors[neighbors != query_i][:k]
        all_indices = [query_i] + list(neighbors)

        n_plots = len(all_indices)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots))
        if n_plots == 1:
            axes = [axes]

        for j, (ax, idx) in enumerate(zip(axes, all_indices)):
            series = data[idx].flatten()
            real_len = int(lengths[idx])

            dist_val = dists[query_i, idx] if idx != query_i else 0.0
            label = "QUERY" if j == 0 else f"Neighbor {j} (dist={dist_val:.4f})"
            color = '#E53935' if j == 0 else '#1E88E5'

            ax.plot(series[:real_len], linewidth=1, color=color, alpha=0.8)
            ax.set_title(f"{label} — Length: {real_len}", fontsize=10)
            ax.set_ylabel("Value")

        axes[-1].set_xlabel("Timestep")
        fig.suptitle(f"Query #{q_idx + 1} — Nearest Neighbors",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(output_dir / f"neighbors_query_{q_idx + 1}.png"), dpi=120)
        plt.close(fig)

    print(f"Saved {n_queries} neighbor plots to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="TS2Vec encoder diagnostics")
    parser.add_argument('--index', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--conf', type=str, default=None)
    parser.add_argument('--n-samples', type=int, default=3000)
    parser.add_argument('--min-length', type=int, default=100)
    parser.add_argument('--n-queries', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='diagnostics')
    args = parser.parse_args()

    pcfg = PipelineConfig(args.conf)
    tcfg = pcfg.training

    # Load index
    index_path = Path(args.index)
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    index = load_index(index_path)

    # Load model
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model = TS2Vec(input_dim=1, hidden_dim=tcfg["hidden_dim"],
                   repr_dim=tcfg["repr_dim"], depth=tcfg["depth"])
    model.load(str(model_path))

    # Output dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect and encode
    series_list, lengths, benchmarks = collect_samples(
        index, args.n_samples, args.min_length)
    data, embeddings = preprocess_and_encode(model, series_list)

    # t-SNE
    print("Running t-SNE...")
    coords = TSNE(n_components=2, perplexity=30,
                  random_state=42, max_iter=1000).fit_transform(embeddings)

    plot_tsne_length(coords, lengths, output_dir / "tsne_length.png")
    plot_tsne_benchmark(coords, benchmarks, output_dir / "tsne_benchmark.png")

    # Nearest neighbors
    plot_neighbors(data, embeddings, lengths, output_dir,
                   n_queries=args.n_queries)

    print(f"\nDiagnostics complete: {output_dir}/")


if __name__ == "__main__":
    main()
