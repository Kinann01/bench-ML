#!/usr/bin/env python3
"""
Classify configs as 'long' or 'short' based on post-SSD sequence length.

For each config in the input JSON, looks up ONE matching benchmark run
from the pre-built index pickle, applies the SteadyStateDetector, and
checks if the post-cutoff length is >= 100 (long) or < 100 (short).

Usage:
    python3 classify_configs.py \
        --configs configs/configs_verified.json \
        --index 2020_index.pkl

Outputs:
    - Prints classification for every config
    - Saves long configs to configs/configs_long.json
    - Saves short configs to configs/configs_short.json
"""

import json
import pickle
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from detector import SteadyStateDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

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


# ---------------------------------------------------------------------------
#  Index loading
# ---------------------------------------------------------------------------

def load_index(path: str) -> Dict[tuple, List[Tuple[Path, int]]]:
    """Load pre-built Config → [(run_dir, version)] index."""
    with open(path, 'rb') as f:
        raw = pickle.load(f)

    result = {}
    for key, entries in raw.items():
        result[key] = [(Path(p), version) for p, version in entries]

    return result


# ---------------------------------------------------------------------------
#  Measure post-SSD length
# ---------------------------------------------------------------------------

_detector = SteadyStateDetector()

MAX_SAMPLES = 10  # Max runs to sample per config for median calculation


def _measure_one(run_dir: Path) -> Tuple[Optional[int], str]:
    """
    Read CSV, apply SSD, return (post-cutoff length, reason).
    Returns (None, reason) on failure.
    """
    csv_path = _find_csv(run_dir)
    if csv_path is None:
        return None, 'no_csv'

    try:
        df = pd.read_csv(csv_path)
        iter_col = _resolve_iter_col(df)
        if df.empty or iter_col is None:
            return None, 'no_col'

        cutoff_idx = _detector.detect_cutoff_index(df)
        if cutoff_idx == 0:
            return None, 'ssd_zero'

        return len(df) - cutoff_idx, 'ok'

    except Exception as e:
        return None, 'read_error'


def measure_median_length(entries: List[Tuple[Path, int]]) -> Tuple[Optional[int], int, Dict[str, int]]:
    """
    Sample up to MAX_SAMPLES runs from a config's entries, measure post-SSD
    length for each, and return (median_length, n_measured, failure_reasons).

    Uses median to be robust against occasional short/failed runs.
    Returns (None, 0, reasons) if no runs could be measured.
    """
    # Evenly space samples across the version range
    if len(entries) <= MAX_SAMPLES:
        sample = entries
    else:
        indices = np.linspace(0, len(entries) - 1, MAX_SAMPLES, dtype=int)
        sample = [entries[i] for i in indices]

    lengths = []
    reasons: Dict[str, int] = {}
    for run_dir, version in sample:
        length, reason = _measure_one(run_dir)
        if length is not None:
            lengths.append(length)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1

    if not lengths:
        return None, 0, reasons

    return int(np.median(lengths)), len(lengths), reasons


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Classify configs as long/short")
    parser.add_argument('--configs', type=str, default='configs/configs_verified.json',
                        help='Input JSON with configs')
    parser.add_argument('--index', type=str, default='2020_index.pkl',
                        help='Pre-built index pickle (from build_index.py)')
    parser.add_argument('--threshold', type=int, default=100,
                        help='Length threshold: >= threshold = long (default: 100)')
    parser.add_argument('--output-long', type=str, default='configs/configs_long.json',
                        help='Output JSON for long configs')
    parser.add_argument('--output-short', type=str, default='configs/configs_short.json',
                        help='Output JSON for short configs')
    args = parser.parse_args()

    # Load index
    index_path = Path(args.index)
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    index = load_index(str(index_path))
    logger.info(f"Loaded index with {len(index)} configs from {index_path}")

    # Load configs
    configs_path = Path(args.configs)
    if not configs_path.is_absolute():
        configs_path = ROOT / configs_path
    with open(configs_path) as f:
        config_dicts = json.load(f)

    configs = [Config(**c) for c in config_dicts]
    logger.info(f"Loaded {len(configs)} configs from {configs_path}")

    long_configs = []
    short_configs = []
    failed_configs = []
    failure_categories = {'not_in_index': 0, 'no_csv': 0, 'no_col': 0,
                          'ssd_zero': 0, 'read_error': 0}

    print(f"\n{'='*90}")
    print(f"{'Config':<50} {'Length':>8} {'Class':>8}  {'Detail'}")
    print(f"{'-'*90}")

    for config in configs:
        label = f"{config.benchmark_type} h={config.machine_host} gc={config.gc_config}"

        # Look up config in index
        key = (config.benchmark_type, config.machine_host,
               config.platform_type, config.gc_config)
        entries = index.get(key)

        if not entries:
            print(f"{label:<50} {'N/A':>8} {'FAIL':>8}  NOT IN INDEX")
            failed_configs.append(config)
            failure_categories['not_in_index'] += 1
            continue

        # Measure median length across multiple runs
        median_len, n_measured, fail_reasons = measure_median_length(entries)

        if median_len is None:
            reason_str = ', '.join(f"{k}={v}" for k, v in fail_reasons.items())
            print(f"{label:<50} {'N/A':>8} {'FAIL':>8}  0/{len(entries)} runs ok ({reason_str})")
            failed_configs.append(config)
            for reason, count in fail_reasons.items():
                failure_categories[reason] = failure_categories.get(reason, 0) + count
            continue

        if median_len >= args.threshold:
            cls = "LONG"
            long_configs.append(config)
        else:
            cls = "SHORT"
            short_configs.append(config)

        print(f"{label:<50} {median_len:>8} {cls:>8}  (median of {n_measured})")

    print(f"{'='*90}")
    print(f"\nSummary:")
    print(f"  Long  (>= {args.threshold} timesteps): {len(long_configs)}")
    print(f"  Short (<  {args.threshold} timesteps): {len(short_configs)}")
    if failed_configs:
        print(f"  Failed:                    {len(failed_configs)}")
        print(f"\nFailure breakdown:")
        reason_labels = {
            'not_in_index': 'Config not found in index',
            'no_csv':       'No CSV file (neither default.csv nor .one-per-rep.csv)',
            'no_col':       'CSV missing iteration_time_ns column',
            'ssd_zero':     'SSD returned cutoff=0 (no steady state detected)',
            'read_error':   'CSV read/parse error',
        }
        for reason, count in sorted(failure_categories.items(), key=lambda x: -x[1]):
            if count > 0:
                desc = reason_labels.get(reason, reason)
                print(f"    {count:>5d}  {desc}")
    print()

    # Save outputs
    output_long = Path(args.output_long)
    if not output_long.is_absolute():
        output_long = ROOT / output_long
    output_long.parent.mkdir(parents=True, exist_ok=True)
    with open(output_long, 'w') as f:
        json.dump([asdict(c) for c in long_configs], f, indent=2)
    print(f"Saved {len(long_configs)} long configs  → {output_long}")

    output_short = Path(args.output_short)
    if not output_short.is_absolute():
        output_short = ROOT / output_short
    output_short.parent.mkdir(parents=True, exist_ok=True)
    with open(output_short, 'w') as f:
        json.dump([asdict(c) for c in short_configs], f, indent=2)
    print(f"Saved {len(short_configs)} short configs → {output_short}")


if __name__ == "__main__":
    main()
