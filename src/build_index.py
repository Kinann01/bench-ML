#!/usr/bin/env python3

import pickle
import argparse
from pathlib import Path

from loader import discover_all_runs

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Build persistent Config -> [(path, version)] index")
    parser.add_argument('--base-dir', type=str, nargs='+', required=True,
                        help='Base directories containing measurement/ and metadata/')
    parser.add_argument('--output', type=str, default='run_index.pkl',
                        help='Output pickle file (default: run_index.pkl)')
    args = parser.parse_args()

    all_runs = {}

    for base in args.base_dir:
        base_dir = Path(base).resolve()
        print(f"\nProcessing: {base_dir}")
        runs = discover_all_runs(base_dir)

        for config, entries in runs.items():
            all_runs.setdefault(config, []).extend(entries)

    # Serialize
    serializable = {}
    for config, entries in all_runs.items():
        key = (config.benchmark_type, config.machine_host,
               config.platform_type, config.gc_config)
        serializable[key] = [(str(path), version) for path, version in entries]

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    with open(output_path, 'wb') as f:
        pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_runs = sum(len(v) for v in all_runs.values())
    total_configs = len(all_runs)

    print(f"\nIndex built:")
    print(f"  Configs:   {total_configs}")
    print(f"  Total runs: {total_runs}")
    print(f"  Saved to:  {output_path}")


if __name__ == "__main__":
    main()
