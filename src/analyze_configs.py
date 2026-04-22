#!/usr/bin/env python3

import csv
import json
import pickle
import logging
import argparse
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

import detector
from detector import SteadyStateDetector
from encoder import TS2Vec
from config import Config
from pipeline_config import PipelineConfig

from constants import (ROOT, PAD_VALUE, 
                       resolve_iter_col, find_csv)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_detector = SteadyStateDetector()

def _load_raw_series(csv_path: str) -> Optional[np.ndarray]:

    try:
        df = pd.read_csv(csv_path)
        iter_col = resolve_iter_col(df)

        if df.empty or iter_col is None:
            return None

        cutoff = _detector.detect_cutoff_index(df)
        if cutoff == 0:
            return None

        series = np.asarray(df[iter_col].iloc[cutoff:].values, dtype=float)
        return series if len(series) >= 2 else None

    except Exception:
        return None


def config_dir_name(config: Config) -> str:
    safe_bench = config.benchmark_type.replace("/", "_").replace(".", "_")
    return f"{safe_bench}_h{config.machine_host}_p{config.platform_type}_gc{config.gc_config}"


def load_index(path: str) -> Dict[tuple, List[Tuple[Path, int]]]:
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    return {key: [(Path(p), v) for p, v in entries]
            for key, entries in raw.items()}

def preprocess_runs(entries: List[Tuple[Path, int]]) -> Tuple[np.ndarray, List[int], dict, List[str]]:

    raw_series = []
    valid_versions = []
    lengths = []
    csv_paths = []

    for run_dir, version in entries:
        csv_path = find_csv(run_dir)
        if csv_path is None:
            continue
        try:
            df = pd.read_csv(csv_path)
            iter_col = resolve_iter_col(df)
            if df.empty or iter_col is None:
                continue

            cutoff_idx = _detector.detect_cutoff_index(df)
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

    concat = np.concatenate(raw_series)
    concat_log = np.log(concat)
    median = float(np.median(concat_log))
    q25, q75 = np.percentile(concat_log, [25, 75])
    iqr = max(float(q75 - q25), 1e-10)

    max_len = max(lengths)
    processed = []
    for s in raw_series:
        s_norm = (np.log(s) - median) / iqr
        padded = np.full(max_len, PAD_VALUE)
        padded[:len(s_norm)] = s_norm
        processed.append(padded)

    padded_runs = np.expand_dims(np.array(processed), axis=-1)

    stats = {
        "n_raw_entries": len(entries),
        "n_valid": len(valid_versions),
        "max_len": max_len,
        "min_len": int(min(lengths)),
        "median_len": int(np.median(lengths)),
        "median_log": median,
        "iqr_log": iqr,
    }

    logger.info(f"  Preprocessed {len(processed)} runs -> shape {padded_runs.shape}")
    return padded_runs, valid_versions, stats, csv_paths


def encode_runs(model: TS2Vec, padded_runs: np.ndarray) -> np.ndarray:
    embeddings = model.encode(padded_runs, batch_size=64)
    logger.info(f"  Encoded -> embeddings shape {embeddings.shape}")
    return embeddings


def compute_consecutive_distances(embeddings: np.ndarray) -> np.ndarray:
    n = len(embeddings)
    return np.array([cosine(embeddings[i], embeddings[i + 1])
                     for i in range(n - 1)])


def flag_anomalies(distances: np.ndarray, versions: List[int],
                   threshold_percentile: float = 97.0,
                   min_z_score: float = 2.0) -> Tuple[List[dict], float]:

    if len(distances) < 3:
        return [], 0.0

    threshold = float(np.percentile(distances, threshold_percentile))
    mean_d = float(np.mean(distances))
    std_d = float(np.std(distances))

    flagged = []
    for idx in np.where(distances > threshold)[0]:
        z = (distances[idx] - mean_d) / std_d if std_d > 0 else 0.0
        if z >= min_z_score:
            flagged.append({
                "version_from": versions[idx],
                "version_to": versions[idx + 1],
                "distance": float(distances[idx]),
                "z_score": float(z),
            })

    return flagged, threshold


def score_config_reliability(distances: np.ndarray, prep_stats: dict,
                             flagged: List[dict],
                             cfg: Optional[dict] = None) -> Tuple[str, List[str]]:

    if cfg is None:
        cfg = PipelineConfig().reliability

    warnings = []
    mean_d = float(np.mean(distances)) if len(distances) > 0 else 0.0
    n_versions = prep_stats.get('n_valid', 0)
    max_len = prep_stats.get('max_len', 1)
    min_len = max(prep_stats.get('min_len', 1), 1)
    length_ratio = max_len / min_len

    if mean_d > cfg["mean_dist_threshold"]:
        warnings.append(f"High mean distance ({mean_d:.3f} > {cfg['mean_dist_threshold']})")
    if length_ratio > cfg["length_ratio_threshold"]:
        warnings.append(f"High length range ratio ({length_ratio:.1f}x > {cfg['length_ratio_threshold']}x)")
    if n_versions < cfg["min_versions"]:
        warnings.append(f"Few versions ({n_versions} < {cfg['min_versions']})")
    if flagged:
        max_z = max(f['z_score'] for f in flagged)
        if max_z < cfg["min_max_z_score"]:
            warnings.append(f"Low max z-score ({max_z:.2f} < {cfg['min_max_z_score']})")

    if len(warnings) >= 2:
        rating = "RECOMMEND-SKIP"
    elif len(warnings) == 1:
        rating = "WEAK"
    else:
        rating = "STRONG"

    return rating, warnings



def plot_anomaly_neighbors(embeddings: np.ndarray,
                           versions: List[int], distances: np.ndarray,
                           flagged: List[dict], config: Config,
                           config_dir: Path, csv_paths: List[str]):

    if not flagged or not csv_paths:
        return

    flagged_pairs = {(f["version_from"], f["version_to"]) for f in flagged}
    anomaly_indices = [i for i in range(len(distances))
                       if (versions[i], versions[i + 1]) in flagged_pairs]
    if not anomaly_indices:
        return

    for anom_idx in anomaly_indices:
        v_from, v_to = versions[anom_idx], versions[anom_idx + 1]
        anom_dist = distances[anom_idx]

        pair = [(anom_idx + 1, v_to, "ANOMALY", '#E53935'),
                (anom_idx,     v_from, "PREVIOUS", '#FF9800')]

        fig, axes = plt.subplots(2, 1, figsize=(14, 5))

        for ax, (idx, v, label, color) in zip(axes, pair):
            raw = _load_raw_series(csv_paths[idx]) if idx < len(csv_paths) else None

            if raw is None:
                ax.text(0.5, 0.5, 'CSV not available',
                        transform=ax.transAxes, ha='center', fontsize=12, color='gray')
                continue

            raw_ms = raw / 1e6
            ax.plot(raw_ms, linewidth=1, color=color, alpha=0.8)
            ax.set_title(f"{label}: v{v} — Length: {len(raw)}", fontsize=10)
            ax.set_ylabel('Time (ms)')

        axes[-1].set_xlabel('Iteration')
        fig.suptitle(f"Anomaly v{v_from}->v{v_to} (dist={anom_dist:.4f})\n"
                     f"{config.benchmark_type} h={config.machine_host} "
                     f"p={config.platform_type} gc={config.gc_config}",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(config_dir / f"anomaly_v{v_from}_v{v_to}.png"), dpi=120)
        plt.close(fig)

    logger.info(f"  Saved {len(anomaly_indices)} anomaly plots")


def save_distances_csv(distances: np.ndarray, versions: List[int],
                       flagged: List[dict], config_dir: Path):

    flagged_pairs = {(f["version_from"], f["version_to"]) for f in flagged}

    with open(config_dir / "distances.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["version_from", "version_to", "cosine_distance", "flagged"])
        for i in range(len(distances)):
            is_flagged = "YES" if (versions[i], versions[i + 1]) in flagged_pairs else ""
            writer.writerow([versions[i], versions[i + 1],
                             f"{distances[i]:.6f}", is_flagged])


def write_report(config: Config, prep_stats: dict, distances: np.ndarray,
                 versions: List[int], flagged: List[dict], threshold: float,
                 config_dir: Path, rating: str = "STRONG",
                 reliability_warnings: Optional[List[str]] = None,
                 csv_paths: Optional[List[str]] = None,
                 threshold_percentile: float = 97.0,
                 min_z_score: float = 2.0):

    lines = [
        f"{'='*60}",
        f"CONFIG ANALYSIS REPORT",
        f"{'='*60}",
        f"",
        f"Benchmark:     {config.benchmark_type}",
        f"Machine Host:  {config.machine_host}",
        f"Platform Type: {config.platform_type}",
        f"GC Config:     {config.gc_config}",
        f"",
        f"--- Data ---",
        f"Raw entries in index:   {prep_stats.get('n_raw_entries', 'N/A')}",
        f"Valid after SSD:        {prep_stats.get('n_valid', 'N/A')}",
        f"Sequence length (max):  {prep_stats.get('max_len', 'N/A')}",
        f"Sequence length (min):  {prep_stats.get('min_len', 'N/A')}",
        f"Sequence length (med):  {prep_stats.get('median_len', 'N/A')}",
        f"Version range:          {versions[0]} -> {versions[-1]}",
        f"",
        f"--- Consecutive Distances ---",
        f"N transitions:          {len(distances)}",
        f"Mean distance:          {distances.mean():.6f}",
        f"Std distance:           {distances.std():.6f}",
        f"Min distance:           {distances.min():.6f}",
        f"Max distance:           {distances.max():.6f}",
        f"Threshold:              {threshold:.6f}",
        f"Flagged anomalies:      {len(flagged)}",
    ]

    if len(distances) >= 1:
        mean_d = float(distances.mean())
        std_d = float(distances.std())
        z_scores = ((distances - mean_d) / std_d) if std_d > 0 else np.zeros_like(distances)

        # Max observed z-score
        i_max = int(np.argmax(z_scores))
        lines += [
            f"",
            f"--- Max observed z-score ---",
            f"Max z = {z_scores[i_max]:+.2f}  "
            f"(v{versions[i_max]} -> v{versions[i_max + 1]}, "
            f"dist={distances[i_max]:.6f})",
        ]

        # Flag count at different z-thresholds (gate is z>=z AND d>threshold)
        z_grid = [1.0, 1.5, 2.0, 2.5, 3.0]
        lines += [
            f"",
            f"--- Flag count at different z-thresholds ---",
            f"(transitions where dist > P{threshold_percentile:g} "
            f"AND z >= z_thr)",
        ]
        for z_thr in z_grid:
            n = int(np.sum((distances > threshold) & (z_scores >= z_thr)))
            tag = "  <- current" if abs(z_thr - min_z_score) < 1e-9 else ""
            lines.append(f"  z >= {z_thr:.1f}:  {n} flagged{tag}")

        # Distance percentiles
        pct_grid = [50, 75, 90, 95, 97, 99]
        lines += [
            f"",
            f"--- Distance percentiles ---",
        ]
        for p in pct_grid:
            v = float(np.percentile(distances, p))
            tag = "  <- current gate" if abs(p - threshold_percentile) < 1e-9 else ""
            lines.append(f"  P{p:>2d}:  {v:.6f}{tag}")

        # Top-K transitions by distance (incl. non-flagged)
        flagged_pairs = {(f_["version_from"], f_["version_to"]) for f_ in flagged}
        k = min(5, len(distances))
        order = np.argsort(distances)[::-1][:k]
        lines += [
            f"",
            f"--- Top {k} transitions by distance ---",
        ]
        for rank, idx in enumerate(order, start=1):
            v_from, v_to = versions[idx], versions[idx + 1]
            mark = " [FLAGGED]" if (v_from, v_to) in flagged_pairs else ""
            lines.append(
                f"  #{rank}: v{v_from} -> v{v_to}  "
                f"dist={distances[idx]:.6f}  z={z_scores[idx]:+.2f}{mark}")

    lines += [
        f"",
        f"--- Reliability ---",
        f"Rating:                 {rating}",
    ]

    for w in (reliability_warnings or []):
        lines.append(f"  WARNING: {w}")

    if flagged:
        lines.append(f"")
        lines.append(f"--- Flagged Anomalies ---")
        for f_ in flagged:
            lines.append(f"  v{f_['version_from']} -> v{f_['version_to']}: "
                         f"dist={f_['distance']:.6f}, z={f_['z_score']:.2f}")

    if csv_paths and len(csv_paths) == len(versions):
        # A version is "flagged" if it appears as either endpoint of any
        # flagged transition.
        flagged_versions = set()
        for f_ in flagged:
            flagged_versions.add(f_["version_from"])
            flagged_versions.add(f_["version_to"])
        lines.append(f"")
        lines.append(f"--- Measurement CSV Paths "
                     f"(version : flagged? : path) ---")
        for v, p in zip(versions, csv_paths):
            mark = "YES" if v in flagged_versions else "NO "
            lines.append(f"  v{v} : {mark} : {p}")

    lines.append(f"{'='*60}")

    with open(config_dir / "report.txt", 'w') as f:
        f.write('\n'.join(lines) + '\n')


def cross_config_analysis(results_summary: List[dict],
                          cfg: Optional[dict] = None) -> List[dict]:

    if cfg is None:
        cfg = PipelineConfig().cross_config

    groups = defaultdict(list)
    configs_per_benchmark = defaultdict(set)

    for entry in results_summary:
        c = entry["config"]
        bench, platform = c["benchmark_type"], c["platform_type"]
        host, gc = c["machine_host"], c["gc_config"]

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

    confidence_order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2,
                        "SINGLE-CONFIG": 3, "N/A": 4}

    report = []
    for (bench, platform, v_from, v_to), hits in sorted(groups.items()):
        n_total = len(configs_per_benchmark[(bench, platform)])
        n_flagged = len(hits)
        ratio = n_flagged / n_total if n_total > 0 else 0.0

        if n_total <= 1:
            confidence = "N/A"
        elif ratio >= cfg["strong_ratio"]:
            confidence = "STRONG"
        elif ratio >= cfg["moderate_ratio"] and n_flagged >= cfg["min_flagged_for_moderate"]:
            confidence = "MODERATE"
        elif n_flagged >= cfg["min_flagged_for_weak"]:
            confidence = "WEAK"
        else:
            confidence = "SINGLE-CONFIG"

        gcs_flagged = sorted(set(h["gc_config"] for h in hits))
        hosts_flagged = sorted(set(h["machine_host"] for h in hits))

        report.append({
            "benchmark_type": bench,
            "platform_type": platform,
            "version_from": v_from,
            "version_to": v_to,
            "n_configs_flagged": n_flagged,
            "n_configs_total": n_total,
            "ratio": round(ratio, 3),
            "confidence": confidence,
            "gcs_flagged": gcs_flagged,
            "hosts_flagged": hosts_flagged,
        })

    report.sort(key=lambda r: (confidence_order[r["confidence"]], -r["ratio"]))
    return report


def main():
    parser = argparse.ArgumentParser(description="Per-config anomaly analysis pipeline")
    parser.add_argument('--configs', type=str, required=True)
    parser.add_argument('--index', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--conf', type=str, default=None,
                        help='Path to pipeline.conf (uses defaults if omitted)')
    parser.add_argument('--output-dir', type=str, default='reports')
    parser.add_argument('--bname', type=str, default=None,
                        help="If set, only process configs whose "
                             "benchmark_type matches this name exactly.")
    args = parser.parse_args()

    # Load pipeline config
    pcfg = PipelineConfig(args.conf)
    detector.configure(pcfg.ssd)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load index
    index_path = Path(args.index)
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    raw_index = load_index(str(index_path))
    logger.info(f"Loaded index: {len(raw_index)} configs")

    # Load configs
    configs_path = Path(args.configs)
    if not configs_path.is_absolute():
        configs_path = ROOT / configs_path
    with open(configs_path) as f:
        configs = [Config(**c) for c in json.load(f)]

    logger.info(f"Loaded {len(configs)} configs")

    if args.bname:
        before = len(configs)
        configs = [c for c in configs if c.benchmark_type == args.bname]
        logger.info(f"Filtered to benchmark '{args.bname}': "
                    f"{len(configs)}/{before} configs")
        if not configs:
            logger.warning(f"No configs match benchmark '{args.bname}'. "
                           f"Exiting.")
            return

    training_cfg = pcfg.training

    # Load model
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model = TS2Vec(input_dim=1, hidden_dim=training_cfg["hidden_dim"], repr_dim=training_cfg["repr_dim"], depth=training_cfg["depth"])
    model.load(str(model_path))

    # Process
    results_summary = []

    for i, config in enumerate(configs):
        logger.info(f"\nConfig {i+1}/{len(configs)}: {config.benchmark_type} "
                     f"h={config.machine_host} p={config.platform_type} gc={config.gc_config}")

        key = (config.benchmark_type, config.machine_host,
               config.platform_type, config.gc_config)
        entries = raw_index.get(key, [])
        if len(entries) < 2:
            logger.warning(f"  Skipping: only {len(entries)} runs")
            continue

        config_dir = output_dir / config_dir_name(config)
        config_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_dir / "config.json", 'w') as f:
                json.dump(asdict(config), f, indent=2)

            padded_runs, versions, prep_stats, csv_paths = preprocess_runs(entries)
            if len(versions) < 2:
                logger.warning(f"  Skipping: only {len(versions)} valid runs after SSD")
                continue

            embeddings = encode_runs(model, padded_runs)
            distances = compute_consecutive_distances(embeddings)

            flagged, threshold = flag_anomalies(
                distances, versions,
                threshold_percentile=pcfg.anomaly_flagging["threshold_percentile"],
                min_z_score=pcfg.anomaly_flagging["min_z_score"])

            plot_anomaly_neighbors(embeddings, versions, distances,
                                   flagged, config, config_dir, csv_paths)
            save_distances_csv(distances, versions, flagged, config_dir)

            rating, warnings = score_config_reliability(
                distances, prep_stats, flagged, cfg=pcfg.reliability)

            write_report(config, prep_stats, distances, versions,
                         flagged, threshold, config_dir,
                         rating=rating, reliability_warnings=warnings,
                         csv_paths=csv_paths,
                         threshold_percentile=pcfg.anomaly_flagging["threshold_percentile"],
                         min_z_score=pcfg.anomaly_flagging["min_z_score"])

            logger.info(f"  Rating: {rating} | Anomalies: {len(flagged)}")

            results_summary.append({
                "config": asdict(config),
                "dir": config_dir_name(config),
                "n_versions": len(versions),
                "n_anomalies": len(flagged),
                "mean_distance": float(distances.mean()),
                "threshold": threshold,
                "rating": rating,
                "flagged_transitions": flagged,
            })

        except Exception as e:
            logger.error(f"  Failed: {e}")

    # Save summary
    with open(output_dir / "analysis_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)

    cross_report = cross_config_analysis(results_summary, cfg=pcfg.cross_config)

    if cross_report:
        cross_csv = output_dir / "cross_config_report.csv"
        with open(cross_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["benchmark_type", "platform_type", "version_from",
                             "version_to", "n_flagged", "n_total", "ratio",
                             "confidence", "gcs_flagged", "hosts_flagged"])
            for r in cross_report:
                writer.writerow([r["benchmark_type"], r["platform_type"],
                                 r["version_from"], r["version_to"],
                                 r["n_configs_flagged"], r["n_configs_total"],
                                 r["ratio"], r["confidence"],
                                 ";".join(str(g) for g in r["gcs_flagged"]),
                                 ";".join(str(h) for h in r["hosts_flagged"])])
        logger.info(f"  Cross-config report: {cross_csv} ({len(cross_report)} transitions)")

    logger.info(f"\nDone. {len(results_summary)}/{len(configs)} configs")


if __name__ == "__main__":
    main()
