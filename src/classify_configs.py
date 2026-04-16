#!/usr/bin/env python3

import json
import pickle
import argparse
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from detector import SteadyStateDetector
from config import Config


ROOT = Path(__file__).resolve().parent.parent

CSV_FILENAMES = ['default.csv.one-per-rep.csv', 'default.csv']
ITERATION_TIME_COLS = ['pol_dd_0_iteration_time_ns', 'iteration_time_ns']


def _find_csv(run_dir: Path) -> Optional[Path]:
    for name in CSV_FILENAMES:
        p = run_dir / name
        if p.exists():
            return p
    return None


def _resolve_iter_col(df: pd.DataFrame) -> Optional[str]:
    for col in ITERATION_TIME_COLS:
        if col in df.columns:
            return col
    return None

def load_index(path: Path) -> Dict[tuple, List[Tuple[Path, int]]]:

    with open(path, 'rb') as f:
        raw = pickle.load(f)

    result = {}
    for key, entries in raw.items():
        result[key] = [(Path(p), version) for p, version in entries]
    return result

_detector = SteadyStateDetector()
MAX_SAMPLES = 10

def _measure_one(run_dir: Path) -> Optional[int]:

    csv_path = _find_csv(run_dir)

    if csv_path is None:
        return None
    try:

        df = pd.read_csv(csv_path)

        if df.empty or _resolve_iter_col(df) is None:
            return None
        cutoff_idx = _detector.detect_cutoff_index(df)
        if cutoff_idx == 0:
            return None
        return len(df) - cutoff_idx
    except Exception:
        return None

def measure_median_length(entries: List[Tuple[Path, int]]) -> Optional[int]:

    if len(entries) <= MAX_SAMPLES:
        sample = entries
    else:
        indices = np.linspace(0, len(entries) - 1, MAX_SAMPLES, dtype=int)
        sample = [entries[i] for i in indices]

    lengths = []
    for run_dir, _ in sample:
        length = _measure_one(run_dir)
        if length is not None:
            lengths.append(length)

    if not lengths:
        return None

    return int(np.median(lengths))


def main():
    parser = argparse.ArgumentParser(
        description="Generate configs JSON from index, filtered by min sequence length")
    parser.add_argument('--index', type=str, required=True,
                        help='Pre-built index pickle (from build_index.py)')
    parser.add_argument('--min-length', type=int, default=100,
                        help='Minimum median post-SSD length (default: 100)')
    parser.add_argument('--output', type=str, default='configs/configs.json',
                        help='Output JSON file (default: configs/configs.json)')
    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    index = load_index(index_path)
    print(f"Loaded index: {len(index)} configs")

    accepted = []
    rejected = 0
    failed = 0

    print(f"\n{'='*80}")
    print(f"{'Config':<55} {'Length':>8} {'Status':>8}")
    print(f"{'-'*80}")

    for key, entries in sorted(index.items()):

        config = Config(benchmark_type=key[0], machine_host=key[1],
                        platform_type=key[2], gc_config=key[3])
        
        label = (f"{config.benchmark_type} h={config.machine_host} "
                 f"p={config.platform_type} gc={config.gc_config}")

        median_len = measure_median_length(entries)

        if median_len is None:
            print(f"{label:<55} {'N/A':>8} {'FAIL':>8}")
            failed += 1
        elif median_len >= args.min_length:
            print(f"{label:<55} {median_len:>8} {'OK':>8}")
            accepted.append(config)
        else:
            print(f"{label:<55} {median_len:>8} {'SKIP':>8}")
            rejected += 1

    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  Accepted (>= {args.min_length}): {len(accepted)}")
    print(f"  Rejected (<  {args.min_length}): {rejected}")
    print(f"  Failed (no data):      {failed}")
    print(f"  Total in index:        {len(index)}")

    output_path = Path(args.output)

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump([asdict(c) for c in accepted], f, indent=2)

    print(f"\nSaved {len(accepted)} configs -> {output_path}")


if __name__ == "__main__":
    main()
