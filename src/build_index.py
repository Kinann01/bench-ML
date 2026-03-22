#!/usr/bin/env python3
"""
Build a persistent index: Config → [(run_dir, version)] mapping.

Scans all year directories once and saves the result as a pickle file.
This avoids re-traversing measurement directories on every analysis run.

Usage:
    python3 build_index.py --base-dir /path/to/data
    python3 build_index.py --base-dir /path/to/data --output my_index.pkl
"""

import pickle
import argparse
import logging
from pathlib import Path

from loader import discover_all_runs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Build persistent Config → runs index")
    parser.add_argument('--base-dir', type=str, default=str(ROOT),
                        help='Base directory containing year folders (default: script dir)')
    parser.add_argument('--output', type=str, default='run_index.pkl',
                        help='Output pickle file (default: run_index.pkl)')
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    output_path = ROOT / args.output

    logger.info(f"Base directory: {base_dir}")

    # Single scan — build full mapping
    all_runs = discover_all_runs(base_dir)

    # Save to pickle
    # Convert Path keys to strings for portability
    serializable = {}
    for config, entries in all_runs.items():
        key = (config.benchmark_type, config.machine_host,
               config.platform_type, config.gc_config)
        serializable[key] = [(str(path), version) for path, version in entries]

    with open(output_path, 'wb') as f:
        pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Summary
    total_runs = sum(len(v) for v in all_runs.values())
    total_configs = len(all_runs)

    logger.info(f"\nIndex built successfully!")
    logger.info(f"  Configs: {total_configs}")
    logger.info(f"  Total runs: {total_runs}")
    logger.info(f"  Saved to: {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Print top benchmarks
    from collections import Counter
    bench_counts = Counter(c.benchmark_type for c in all_runs.keys())
    logger.info(f"\nTop benchmarks by config count:")
    for bench, count in bench_counts.most_common(10):
        logger.info(f"  {bench}: {count} configs")


def load_index(path: str) -> dict:
    """
    Load a saved index and return raw mapping.

    Returns:
        Dict with tuple keys (benchmark_type, machine_host, platform_type, gc_config)
        mapped to lists of (Path, version_int) tuples.

    The caller should construct their own Config objects from the tuple keys
    to avoid __main__ vs module import identity issues.

    Usage:
        raw = load_index('run_index.pkl')
        for (bench, host, ptype, gc), entries in raw.items():
            config = Config(bench, host, ptype, gc)
            ...
    """
    with open(path, 'rb') as f:
        serializable = pickle.load(f)

    result = {}
    for key, entries in serializable.items():
        result[key] = [(Path(p), version) for p, version in entries]

    return result


if __name__ == "__main__":
    main()