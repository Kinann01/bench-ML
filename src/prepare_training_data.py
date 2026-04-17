#!/usr/bin/env python3

import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from detector import SteadyStateDetector
from config import Config

from constants import (ROOT, PAD_VALUE,
                    find_csv, resolve_iter_col)

_detector = SteadyStateDetector()

def load_index(path: Path) -> Dict[tuple, List[Tuple[Path, int]]]:
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    return {key: [(Path(p), v) for p, v in entries]
            for key, entries in raw.items()}


def extract_series(entries: List[Tuple[Path, int]]) -> List[np.ndarray]:

    series_list = []

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
            if len(series) >= 2:
                series_list.append(series)

        except Exception:
            continue

    return series_list


def normalize_config(series_list: List[np.ndarray]) -> List[np.ndarray]:

    concat = np.concatenate(series_list)
    concat_log = np.log(concat)
    median = float(np.median(concat_log))
    q25, q75 = np.percentile(concat_log, [25, 75])
    iqr = max(float(q75 - q25), 1e-10)

    return [(np.log(s) - median) / iqr for s in series_list]


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from index: SSD + log+IQR normalization + padding")
    parser.add_argument('--index', type=str, required=True)
    parser.add_argument('--output', type=str, default='data/training_data.npy')
    parser.add_argument('--min-length', type=int, default=2,
                        help='Minimum post-SSD length to include (default: 2)')
    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    index = load_index(index_path)
    print(f"Loaded index: {len(index)} configs")

    all_normalized = []
    all_lengths = []

    for i, (key, entries) in enumerate(sorted(index.items())):
        config = Config(*key)

        raw_series = extract_series(entries)
        if not raw_series:
            continue

        # Filter by min length
        raw_series = [s for s in raw_series if len(s) >= args.min_length]
        if not raw_series:
            continue

        normalized = normalize_config(raw_series)
        all_normalized.extend(normalized)
        all_lengths.extend(len(s) for s in normalized)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(index)} configs, "
                  f"{len(all_normalized)} series so far")

    if not all_normalized:
        print("No valid series found!")
        return

    # Pad to common length
    max_len = max(all_lengths)
    padded = []
    for s in all_normalized:
        p = np.full(max_len, PAD_VALUE)
        p[:len(s)] = s
        padded.append(p)

    data = np.expand_dims(np.array(padded), axis=-1)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(output_path), data)

    print(f"\nTraining data prepared:")
    print(f"  Shape: {data.shape}")
    print(f"  Configs: {len(index)}")
    print(f"  Series: {len(all_normalized)}")
    print(f"  Max length: {max_len}")
    print(f"  Min length: {min(all_lengths)}")
    print(f"  Median length: {int(np.median(all_lengths))}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
