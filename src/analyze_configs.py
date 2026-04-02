#!/usr/bin/env python3
"""
Per-Config Anomaly Analysis Pipeline.

Given a JSON of configs and a pre-built index, for each config:
  1. Load runs from index
  2. Preprocess: SSD cutoff → per-config log+IQR normalization → pad to max_len
  3. Encode with trained TS2Vec → instance-level embeddings
  4. Generate per-config report directory containing:
     - config.json          — config metadata
     - tsne.png             — t-SNE colored by version order
     - distances.csv        — table of (version_from, version_to, distance, flagged)
     - report.txt           — text summary with stats
     - anomaly*.png         — for each flagged anomaly, plot query + neighbors (normalized and raw)

Usage:
    python3 analyze_configs.py \
        --configs configs/configs_long.json \
        --index 2020_index.pkl \
        --model TS2Vec/ts2vec_long.pt \
        --output-dir reports
"""

import csv
import json
import pickle
import logging
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from detector import SteadyStateDetector
from encoder import TS2Vec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

PAD_VALUE = -999.0

# Flexible naming: try each in order
CSV_FILENAMES = ['default.csv.one-per-rep.csv', 'default.csv']
ITERATION_TIME_COLS = ['pol_dd_0_iteration_time_ns', 'iteration_time_ns']


def _find_csv(run_dir: Path) -> Optional[Path]:
    """Return the first existing CSV file from known name variants, or None."""
    for name in CSV_FILENAMES:
        p = run_dir / name
        if p.exists():
            return p
    return None

def _resolve_iter_col(df: pd.DataFrame) -> Optional[str]:
    """Return the iteration time column name that exists in df, or None."""
    for col in ITERATION_TIME_COLS:
        if col in df.columns:
            return col
    return None


@dataclass(frozen=True)
class Config:
    benchmark_type: str
    machine_host: int
    platform_type: int
    gc_config: int


def config_dir_name(config: Config) -> str:
    """Generate a filesystem-safe directory name for a config."""
    safe_bench = config.benchmark_type.replace("/", "_").replace(".", "_")
    return f"{safe_bench}_h{config.machine_host}_p{config.platform_type}_gc{config.gc_config}"



def _attach_config_log_handler(log_path: Path) -> logging.FileHandler:
    """Attach a file handler to the root logger writing to log_path."""
    handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    return handler


def _detach_log_handler(handler: logging.FileHandler) -> None:
    """Flush, close, and remove a file handler from the root logger."""
    handler.flush()
    handler.close()
    logging.getLogger().removeHandler(handler)


def load_index(path: str) -> Dict[tuple, List[Tuple[Path, int]]]:
    """Load pre-built Config → [(run_dir, version)] index."""
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    result = {}
    for key, entries in raw.items():
        result[key] = [(Path(p), version) for p, version in entries]
    return result


def preprocess_runs(entries: List[Tuple[Path, int]]) -> Tuple[np.ndarray, List[int], dict, List[str]]:
    """
    Load, apply SSD, log+IQR normalize, and pad all runs for one config.

    Returns:
        X: shape (n_runs, max_len, 1) — preprocessed, padded data
        versions: sorted list of version ints matching row order
        stats: dict with preprocessing statistics
        csv_paths: list of CSV paths corresponding to each run (for anomaly neighbor plots)
    """
    detector = SteadyStateDetector()

    raw_series = []
    valid_versions = []
    lengths = []
    csv_paths = []

    for run_dir, version in entries:
        csv_path = _find_csv(run_dir)
        if csv_path is None:
            continue
        try:
            df = pd.read_csv(csv_path)
            iter_col = _resolve_iter_col(df)
            if df.empty or iter_col is None:
                continue
            cutoff_idx = detector.detect_cutoff_index(df)
            if cutoff_idx == 0:
                continue

            series = df[iter_col].iloc[cutoff_idx:].values.astype(float)
            if len(series) < 2:
                continue

            raw_series.append(series)
            valid_versions.append(version)
            lengths.append(len(series))
            csv_paths.append(str(csv_path))
        except Exception as e:
            logger.warning(f"  Failed to load {csv_path}: {e}")

    if not raw_series:
        return np.array([]), [], {}, []

    # Per-config log + IQR normalization
    concat = np.concatenate(raw_series)
    concat_log = np.log(concat)
    median = float(np.median(concat_log))
    q25, q75 = np.percentile(concat_log, [25, 75])
    iqr = float(q75 - q25)
    if iqr < 1e-10:
        iqr = 1.0

    # Log-transform, normalize, and pad
    max_len = max(len(s) for s in raw_series)
    processed = []
    for s in raw_series:
        s_norm = (np.log(s) - median) / iqr
        padded = np.full(max_len, PAD_VALUE)
        padded[:len(s_norm)] = s_norm
        processed.append(padded)

    X = np.array(processed)
    X = np.expand_dims(X, axis=-1)  # (n_runs, max_len, 1)

    stats = {
        "n_raw_entries": len(entries),
        "n_valid": len(valid_versions),
        "max_len": max_len,
        "min_len": int(min(lengths)),
        "median_len": int(np.median(lengths)),
        "mean_raw": float(np.mean(concat)),
        "std_raw": float(np.std(concat)),
        "median_log": median,
        "iqr_log": iqr,
    }

    logger.info(f"  Preprocessed {len(processed)} runs → X shape {X.shape}")
    return X, valid_versions, stats, csv_paths


def encode_runs(model: TS2Vec, X: np.ndarray) -> np.ndarray:
    """Encode preprocessed runs → instance-level embeddings."""
    embeddings = model.encode(X, batch_size=64)
    logger.info(f"  Encoded → embeddings shape {embeddings.shape}")
    return embeddings


def compute_consecutive_distances(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance between each pair of consecutive embeddings."""
    from scipy.spatial.distance import cosine
    n = len(embeddings)
    distances = np.zeros(n - 1)
    for i in range(n - 1):
        distances[i] = cosine(embeddings[i], embeddings[i + 1])
    return distances


def plot_tsne(embeddings: np.ndarray, versions: List[int],
              config: Config, config_dir: Path):
    """t-SNE plot colored by version order, saved to config_dir/tsne.png."""
    from sklearn.manifold import TSNE

    n_samples = len(embeddings)
    if n_samples < 2:
        logger.warning(f"  Skipping t-SNE: only {n_samples} samples")
        return

    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    order = np.arange(n_samples)

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=order, cmap='coolwarm',
                    s=30, alpha=0.8, edgecolors='black', linewidth=0.3)
    plt.colorbar(sc, ax=ax, label='Version Order (early → late)')

    # Label key points
    label_indices = [0, n_samples - 1]
    if n_samples > 5:
        label_indices += [n_samples // 4, n_samples // 2, 3 * n_samples // 4]

    for idx in label_indices:
        ax.annotate(f"v{versions[idx]}",
                    (coords[idx, 0], coords[idx, 1]),
                    fontsize=7, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points')

    ax.set_title(f"t-SNE: {config.benchmark_type}\n"
                 f"host={config.machine_host}, platform={config.platform_type}, "
                 f"gc={config.gc_config} — {n_samples} commits",
                 fontsize=11)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(str(config_dir / "tsne.png"), dpi=120)
    plt.close(fig)


def flag_anomalies(distances: np.ndarray, versions: List[int],
                   threshold_percentile: float = 97.0,
                   min_z_score: float = 2.0) -> Tuple[List[dict], float]:
    """
    Compute anomaly threshold and flag consecutive distances above it.

    A transition is only flagged if it exceeds both the percentile threshold
    AND has a z-score >= min_z_score relative to this config's distance distribution.
    This prevents flagging transitions that are statistically at the tail but not
    meaningfully separated from the noise floor.

    Returns:
        flagged: list of dicts with version_from, version_to, distance, z_score
        threshold: the percentile threshold value
    """
    n_distances = len(distances)
    if n_distances < 3:
        return [], 0.0

    threshold = float(np.percentile(distances, threshold_percentile))
    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))

    anomaly_indices = np.where(distances > threshold)[0]

    flagged = []
    for idx in anomaly_indices:
        z = float((distances[idx] - mean_dist) / std_dist) if std_dist > 0 else 0.0
        if z >= min_z_score:
            flagged.append({
                "version_from": versions[idx],
                "version_to": versions[idx + 1],
                "distance": float(distances[idx]),
                "z_score": z,
            })

    return flagged, threshold


def save_distances_csv(distances: np.ndarray, versions: List[int],
                       flagged: List[dict], config_dir: Path):
    """Save a CSV table of consecutive-commit distances."""
    flagged_pairs = {(f["version_from"], f["version_to"]) for f in flagged}
    csv_path = config_dir / "distances.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["version_from", "version_to", "cosine_distance", "flagged"])
        for i in range(len(distances)):
            is_flagged = "YES" if (versions[i], versions[i + 1]) in flagged_pairs else ""
            writer.writerow([
                versions[i],
                versions[i + 1],
                f"{distances[i]:.6f}",
                is_flagged,
            ])


def plot_anomaly_neighbors(X: np.ndarray, embeddings: np.ndarray,
                           versions: List[int], distances: np.ndarray,
                           flagged: List[dict], config: Config,
                           config_dir: Path, csv_paths: Optional[List[str]] = None,
                           k: int = 5):
    """
    For each flagged anomaly, plot the anomalous version's raw time series
    (red) alongside its K nearest neighbors in embedding space (blue).

    Generates two plots per anomaly:
      - anomaly_vXXX_vYYY.png      — log+IQR normalized data
      - anomaly_vXXX_vYYY_raw.png  — raw iteration time (ns) from CSV

    The query is version[idx+1] (the "new" version that caused the spike).
    """
    from scipy.spatial.distance import cdist

    if not flagged:
        return

    # Build anomaly indices from the flagged list (which already passed both
    # percentile and z-score filters)
    flagged_pairs = {(f["version_from"], f["version_to"]) for f in flagged}
    anomaly_indices = [
        i for i in range(len(distances))
        if (versions[i], versions[i + 1]) in flagged_pairs
    ]
    if not anomaly_indices:
        return

    # Pairwise distance matrix
    dists = cdist(embeddings, embeddings, metric='cosine')

    # Compute real lengths per sample (non-PAD)
    data = X.squeeze(-1)  # (n_runs, max_len)
    real_lengths = np.array([int(np.sum(data[i] != PAD_VALUE)) for i in range(len(data))])

    detector = SteadyStateDetector()
    raw_data = {}  # idx -> raw series (ns), loaded lazily

    def _load_raw(idx):
        """Load raw post-SSD iteration time for a given index."""
        if idx in raw_data:
            return raw_data[idx]
        if csv_paths and idx < len(csv_paths):
            try:
                df = pd.read_csv(csv_paths[idx])
                iter_col = _resolve_iter_col(df)
                if iter_col is None:
                    return None
                cutoff = detector.detect_cutoff_index(df)
                series = df[iter_col].iloc[cutoff:].values.astype(float)
                raw_data[idx] = series
                return series
            except Exception:
                pass
        return None

    for anom_idx in anomaly_indices:
        query_i = anom_idx + 1
        v_from = versions[anom_idx]
        v_to = versions[anom_idx + 1]
        anom_dist = distances[anom_idx]

        # Find K nearest neighbors (excluding the query itself)
        neighbor_order = np.argsort(dists[query_i])
        neighbor_order = neighbor_order[neighbor_order != query_i][:k]

        # Also include the previous version (v_from) for comparison
        all_indices = [query_i, anom_idx] + list(neighbor_order)

        # --- Plot 1: Normalized data ---
        n_plots = len(all_indices)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots))
        if n_plots == 1:
            axes = [axes]

        for j, (ax, idx) in enumerate(zip(axes, all_indices)):
            series = data[idx]
            real_len = int(real_lengths[idx])

            if j == 0:
                label = f"ANOMALY: v{v_to} (query)"
                color = '#E53935'
            elif j == 1:
                prev_dist = dists[query_i, idx]
                label = f"PREVIOUS: v{v_from} (cosine dist={prev_dist:.4f})"
                color = '#FF9800'
            else:
                neighbor_dist = dists[query_i, idx]
                label = f"Neighbor {j-1}: v{versions[idx]} (cosine dist={neighbor_dist:.4f})"
                color = '#1E88E5'

            ax.plot(series[:real_len], linewidth=1, color=color, alpha=0.8)
            ax.set_title(f"{label} \u2014 Length: {real_len}", fontsize=10)
            ax.set_ylabel('normed')

        axes[-1].set_xlabel('Timestep')
        fig.suptitle(f"Anomaly v{v_from}\u2192v{v_to} (dist={anom_dist:.4f}) [normalized]\n"
                     f"{config.benchmark_type} h={config.machine_host} "
                     f"p={config.platform_type} gc={config.gc_config}",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(config_dir / f"anomaly_v{v_from}_v{v_to}.png"), dpi=120)
        plt.close(fig)

        # --- Plot 2: Raw data (iteration time ns) ---
        if csv_paths:
            fig2, axes2 = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots))
            if n_plots == 1:
                axes2 = [axes2]

            for j, (ax, idx) in enumerate(zip(axes2, all_indices)):
                raw_series = _load_raw(idx)
                if raw_series is None:
                    ax.text(0.5, 0.5, 'CSV not available', transform=ax.transAxes,
                            ha='center', va='center', fontsize=12, color='gray')
                    continue

                if j == 0:
                    label = f"ANOMALY: v{v_to} (query)"
                    color = '#E53935'
                elif j == 1:
                    prev_dist = dists[query_i, idx]
                    label = f"PREVIOUS: v{v_from} (cosine dist={prev_dist:.4f})"
                    color = '#FF9800'
                else:
                    neighbor_dist = dists[query_i, idx]
                    label = f"Neighbor {j-1}: v{versions[idx]} (cosine dist={neighbor_dist:.4f})"
                    color = '#1E88E5'

                # Convert to milliseconds for readability
                raw_ms = np.asarray(raw_series) / 1e6
                ax.plot(raw_ms, linewidth=1, color=color, alpha=0.8)
                ax.set_title(f"{label} \u2014 Length: {len(raw_series)}", fontsize=10)
                ax.set_ylabel('Time (ms)')

            axes2[-1].set_xlabel('Timestep')
            fig2.suptitle(f"Anomaly v{v_from}\u2192v{v_to} (dist={anom_dist:.4f}) [raw iteration time]\n"
                          f"{config.benchmark_type} h={config.machine_host} "
                          f"p={config.platform_type} gc={config.gc_config}",
                          fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(str(config_dir / f"anomaly_v{v_from}_v{v_to}_raw.png"), dpi=120)
            plt.close(fig2)

    logger.info(f"  Saved {len(anomaly_indices)} anomaly neighbor plots")


def score_config_reliability(distances: np.ndarray, prep_stats: dict,
                             flagged: List[dict]) -> Tuple[str, List[str]]:
    """
    Rate the reliability of this config's anomaly results.

    Checks four warning signals and returns a rating plus a list of warning strings:
      STRONG         — 0 warnings: results are trustworthy
      WEAK           — 1 warning:  interpret with caution
      RECOMMEND-SKIP — 2+ warnings: results are likely dominated by noise

    Warning signals:
      1. High mean distance (> 0.35): baseline is noisy, threshold is inflated
      2. High length range ratio (> 5×): SSD instability, model may cluster by length
      3. Too few versions (< 10): P97 threshold unreliable with small samples
      4. Low max z-score among flagged (< 1.5): anomalies barely stand out from noise
    """
    warnings = []

    mean_dist = float(np.mean(distances)) if len(distances) > 0 else 0.0
    n_versions = prep_stats.get('n_valid', 0)
    max_len = prep_stats.get('max_len', 1)
    min_len = max(prep_stats.get('min_len', 1), 1)
    length_ratio = max_len / min_len

    if mean_dist > 0.35:
        warnings.append(f"High mean distance ({mean_dist:.3f} > 0.35) — baseline is noisy")
    if length_ratio > 5.0:
        warnings.append(f"High length range ratio ({length_ratio:.1f}x > 5x) — possible SSD instability")
    if n_versions < 10:
        warnings.append(f"Few versions ({n_versions} < 10) — percentile threshold unreliable")
    if flagged:
        max_z = max(f['z_score'] for f in flagged)
        if max_z < 1.5:
            warnings.append(f"Low max z-score ({max_z:.2f} < 1.5) — anomalies barely above noise floor")

    n_warnings = len(warnings)
    if n_warnings >= 2:
        rating = "RECOMMEND-SKIP"
    elif n_warnings == 1:
        rating = "WEAK"
    else:
        rating = "STRONG"

    return rating, warnings


def write_report(config: Config, prep_stats: dict, embeddings: np.ndarray,
                 distances: np.ndarray, versions: List[int],
                 flagged: List[dict], threshold: float, config_dir: Path,
                 rating: str = "STRONG", reliability_warnings: Optional[List[str]] = None,
                 csv_paths: Optional[List[str]] = None):
    """Write a human-readable report.txt summarizing the config analysis."""
    from scipy.spatial.distance import pdist

    n_runs = len(versions)
    norms = np.linalg.norm(embeddings, axis=1)

    # Pairwise cosine distances (subsample if too many)
    if n_runs <= 500:
        pairwise = pdist(embeddings, metric='cosine')
    else:
        idx = np.random.RandomState(42).choice(n_runs, 500, replace=False)
        pairwise = pdist(embeddings[idx], metric='cosine')

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"CONFIG ANALYSIS REPORT")
    lines.append(f"{'='*60}")
    lines.append(f"")
    lines.append(f"Benchmark:     {config.benchmark_type}")
    lines.append(f"Machine Host:  {config.machine_host}")
    lines.append(f"Platform Type: {config.platform_type}")
    lines.append(f"GC Config:     {config.gc_config}")
    lines.append(f"")
    lines.append(f"--- Data ---")
    lines.append(f"Raw entries in index:   {prep_stats.get('n_raw_entries', 'N/A')}")
    lines.append(f"Valid after SSD:        {prep_stats.get('n_valid', 'N/A')}")
    lines.append(f"Sequence length (max):  {prep_stats.get('max_len', 'N/A')}")
    lines.append(f"Sequence length (min):  {prep_stats.get('min_len', 'N/A')}")
    lines.append(f"Sequence length (med):  {prep_stats.get('median_len', 'N/A')}")
    lines.append(f"Version range:          {versions[0]} → {versions[-1]}")
    lines.append(f"Median (log ns):        {prep_stats.get('median_log', 0):.4f}")
    lines.append(f"IQR (log ns):           {prep_stats.get('iqr_log', 0):.4f}")
    lines.append(f"Mean (raw ns):          {prep_stats.get('mean_raw', 0):.2f}")
    lines.append(f"Std (raw ns):           {prep_stats.get('std_raw', 0):.2f}")
    lines.append(f"")
    lines.append(f"--- Embeddings ---")
    lines.append(f"Embedding dim:          {embeddings.shape[1]}")
    lines.append(f"L2 norm (mean):         {norms.mean():.4f}")
    lines.append(f"L2 norm (std):          {norms.std():.4f}")
    lines.append(f"Pairwise cosine (mean): {pairwise.mean():.4f}")
    lines.append(f"Pairwise cosine (std):  {pairwise.std():.4f}")
    lines.append(f"")
    lines.append(f"--- Consecutive Distances ---")
    lines.append(f"N transitions:          {len(distances)}")
    lines.append(f"Mean distance:          {distances.mean():.6f}")
    lines.append(f"Std distance:           {distances.std():.6f}")
    lines.append(f"Min distance:           {distances.min():.6f}")
    lines.append(f"Max distance:           {distances.max():.6f}")
    lines.append(f"Threshold:              {threshold:.6f}")
    lines.append(f"Flagged anomalies:      {len(flagged)}")
    lines.append(f"")
    lines.append(f"--- Reliability ---")
    lines.append(f"Rating:                 {rating}")
    for w in (reliability_warnings or []):
        lines.append(f"  WARNING: {w}")
    lines.append(f"")

    if flagged:
        lines.append(f"--- Flagged Anomalies ---")
        for f_ in flagged:
            lines.append(f"  v{f_['version_from']} → v{f_['version_to']}: "
                         f"dist={f_['distance']:.6f}, z={f_['z_score']:.2f}")
        lines.append(f"")

    lines.append(f"--- Files ---")
    lines.append(f"  config.json     — config metadata")
    lines.append(f"  tsne.png        — t-SNE visualization")
    lines.append(f"  distances.csv   — distance table")
    lines.append(f"  report.txt      — this file")
    lines.append(f"{'='*60}")

    if csv_paths:
        lines.append(f"")
        lines.append(f"--- Source CSV Files ---")
        for i, p in enumerate(csv_paths):
            lines.append(f"  v{versions[i]}: {p}")
        lines.append(f"{'='*60}")

    report_path = config_dir / "report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

def cross_config_analysis(results_summary: List[dict], output_dir: Path) -> None:
    """
    Cross-config corroboration: group flagged transitions by
    (benchmark_type, platform_type, version_from, version_to) across
    all machine_host and gc_config values.

    A transition flagged on multiple hosts/GCs is strong evidence of a
    code regression rather than machine noise or GC-specific behavior.

    Writes cross_config_report.json to output_dir.
    """
    from collections import defaultdict

    # Group: (benchmark_type, platform_type, version_from, version_to) →
    #   list of {host, gc, distance, z_score, rating}
    groups = defaultdict(list)
    # Also track total configs per (benchmark_type, platform_type) to compute coverage
    configs_per_benchmark = defaultdict(set)

    for entry in results_summary:
        cfg = entry["config"]
        bench = cfg["benchmark_type"]
        platform = cfg["platform_type"]
        host = cfg["machine_host"]
        gc = cfg["gc_config"]

        configs_per_benchmark[(bench, platform)].add((host, gc))

        for t in entry.get("flagged_transitions", []):
            key = (bench, platform, t["version_from"], t["version_to"])
            groups[key].append({
                "machine_host": host,
                "gc_config": gc,
                "distance": t["distance"],
                "z_score": t["z_score"],
                "rating": entry.get("rating", "UNKNOWN"),
            })

    report = []
    for (bench, platform, v_from, v_to), hits in sorted(groups.items()):
        hosts_flagged = sorted(set(h["machine_host"] for h in hits))
        gcs_flagged = sorted(set(h["gc_config"] for h in hits))
        n_configs_total = len(configs_per_benchmark[(bench, platform)])

        n_flagged = len(hits)
        ratio = n_flagged / n_configs_total if n_configs_total > 0 else 0.0

        if n_configs_total <= 1:
            confidence = "N/A"
        elif ratio >= 0.40:
            confidence = "STRONG"
        elif ratio >= 0.20 and n_flagged >= 2:
            confidence = "MODERATE"
        elif n_flagged >= 2:
            confidence = "WEAK"
        else:
            confidence = "SINGLE-CONFIG"

        report.append({
            "benchmark_type": bench,
            "platform_type": platform,
            "version_from": v_from,
            "version_to": v_to,
            "hosts_flagged": hosts_flagged,
            "gcs_flagged": gcs_flagged,
            "n_configs_flagged": n_flagged,
            "n_configs_total": n_configs_total,
            "ratio": round(ratio, 3),
            "confidence": confidence,
            "details": hits,
        })

    # Sort: STRONG first, then MODERATE, etc., then by ratio descending
    confidence_order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "SINGLE-CONFIG": 3, "N/A": 4}
    report.sort(key=lambda r: (confidence_order[r["confidence"]], -r["ratio"]))

    # Write JSON report
    report_path = output_dir / "cross_config_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Write CSV report
    csv_path = output_dir / "cross_config_report.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "confidence", "benchmark_type", "platform_type",
            "version_from", "version_to",
            "n_configs_flagged", "n_configs_total", "ratio",
            "hosts_flagged", "gcs_flagged",
        ])
        for r in report:
            writer.writerow([
                r["confidence"],
                r["benchmark_type"],
                r["platform_type"],
                r["version_from"],
                r["version_to"],
                r["n_configs_flagged"],
                r["n_configs_total"],
                r["ratio"],
                ";".join(str(h) for h in r["hosts_flagged"]),
                ";".join(str(g) for g in r["gcs_flagged"]),
            ])

    n_strong = sum(1 for r in report if r["confidence"] == "STRONG")
    n_moderate = sum(1 for r in report if r["confidence"] == "MODERATE")
    logger.info(f"Cross-config analysis: {len(report)} transitions — "
                f"{n_strong} STRONG, {n_moderate} MODERATE")
    logger.info(f"Cross-config report: {report_path}")
    logger.info(f"Cross-config CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-config anomaly analysis pipeline")
    parser.add_argument('--configs', type=str, required=True,
                        help='JSON file with configs (from classify_configs.py)')
    parser.add_argument('--index', type=str, required=True,
                        help='Pre-built index pickle (from build_index.py)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained TS2Vec model checkpoint')
    parser.add_argument('--depth', type=int, default=10,
                        help='TS2Vec encoder depth (default: 10 for long model)')
    parser.add_argument('--output-dir', type=str, default='reports',
                        help='Output directory for per-config reports')
    parser.add_argument('--max-configs', type=int, default=None,
                        help='Process at most N configs (for testing)')
    parser.add_argument('--percentile', type=float, default=97.0,
                        help='Percentile threshold for anomaly flagging (default: 97.0)')
    parser.add_argument('--min-z-score', type=float, default=2.0,
                        dest='min_z_score',
                        help='Minimum z-score for a transition to be flagged (default: 2.0)')
    parser.add_argument('--cross-config', action='store_true', default=False,
                        dest='cross_config',
                        help='Run cross-config corroboration analysis after per-config processing')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load index
    index_path = Path(args.index)
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    raw_index = load_index(str(index_path))
    logger.info(f"Loaded index: {len(raw_index)} configs, "
                f"{sum(len(v) for v in raw_index.values())} total runs")

    # Load configs
    configs_path = Path(args.configs)
    if not configs_path.is_absolute():
        configs_path = ROOT / configs_path
    with open(configs_path) as f:
        config_dicts = json.load(f)

    configs = [Config(**c) for c in config_dicts]
    if args.max_configs:
        configs = configs[:args.max_configs]
    logger.info(f"Loaded {len(configs)} target configs from {configs_path}")

    # Load TS2Vec model
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model = TS2Vec(input_dim=1, hidden_dim=128, repr_dim=320, depth=args.depth)
    model.load(str(model_path))
    logger.info(f"Loaded TS2Vec model (depth={args.depth}) from {model_path}")

    # Process each config
    results_summary = []
    t_start = time.time()

    for i, config in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Config {i+1}/{len(configs)}: {config.benchmark_type} "
                     f"(host={config.machine_host}, platform={config.platform_type}, "
                     f"gc={config.gc_config})")
        logger.info(f"{'='*60}")

        # Look up runs from index
        key = (config.benchmark_type, config.machine_host,
               config.platform_type, config.gc_config)
        entries = raw_index.get(key, [])
        if len(entries) < 2:
            logger.warning(f"  Skipping: only {len(entries)} runs in index")
            continue

        logger.info(f"  Found {len(entries)} runs in index")

        # Create per-config output directory
        dir_name = config_dir_name(config)
        config_dir = output_dir / dir_name
        config_dir.mkdir(parents=True, exist_ok=True)

        log_handler = _attach_config_log_handler(config_dir / "analysis.log")
        try:
            # Save config.json
            with open(config_dir / "config.json", 'w') as f:
                json.dump(asdict(config), f, indent=2)

            # Preprocess
            X, versions, prep_stats, csv_paths = preprocess_runs(entries)
            if len(versions) < 2:
                logger.warning(f"  Skipping: only {len(versions)} valid runs after SSD")
                with open(config_dir / "report.txt", 'w') as f:
                    f.write(f"SKIPPED: only {len(versions)} valid runs after SSD preprocessing.\n")
                    f.write(f"Raw entries in index: {len(entries)}\n")
                continue

            # Encode
            embeddings = encode_runs(model, X)

            # t-SNE visualization
            plot_tsne(embeddings, versions, config, config_dir)

            # Compute distances
            distances = compute_consecutive_distances(embeddings)

            # Flag anomalies
            flagged, threshold = flag_anomalies(distances, versions,
                                                threshold_percentile=args.percentile,
                                                min_z_score=args.min_z_score)
            plot_anomaly_neighbors(X, embeddings, versions, distances, flagged,
                                   config, config_dir, csv_paths)

            # Save distances CSV table
            save_distances_csv(distances, versions, flagged, config_dir)

            # Score config reliability
            rating, reliability_warnings = score_config_reliability(distances, prep_stats, flagged)

            # Write report
            write_report(config, prep_stats, embeddings, distances, versions,
                         flagged, threshold, config_dir,
                         rating=rating, reliability_warnings=reliability_warnings,
                         csv_paths=csv_paths)

            logger.info(f"  Report saved to {config_dir}/")

            results_summary.append({
                "config": asdict(config),
                "dir": dir_name,
                "n_versions": len(versions),
                "version_range": [versions[0], versions[-1]],
                "seq_len_max": prep_stats.get("max_len", 0),
                "n_anomalies": len(flagged),
                "mean_distance": float(distances.mean()),
                "threshold": threshold,
                "rating": rating,
                "flagged_transitions": flagged,
            })
        finally:
            _detach_log_handler(log_handler)

    # Save global summary
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Cross-config corroboration (optional)
    if args.cross_config:
        cross_config_analysis(results_summary, output_dir)

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Done. Processed {len(results_summary)}/{len(configs)} configs "
                f"in {elapsed:.1f}s")
    logger.info(f"Summary: {summary_path}")
    logger.info(f"Reports: {output_dir}/")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
