#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import Config
from constants import (ALLOWED_GC_CONFIGS, ALLOWED_MACHINE_HOSTS,
                       ALLOWED_PLATFORM_TYPES, METADATA_SUBDIRS, 
                       CSV_FILENAMES, ITERATION_TIME_COLS)

def _find_csv(run_dir: Path) -> Optional[Path]:
    for name in CSV_FILENAMES:
        p = run_dir / name
        if p.exists():
            return p
    return None


""" Unused helper but might become useful in the future """
def _resolve_iter_col(df: pd.DataFrame) -> Optional[str]:
    for col in ITERATION_TIME_COLS:
        if col in df.columns:
            return col
    return None


class DataLoader:

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._json_cache: Dict[Path, dict] = {}
        self.grouped_data: Dict[Config, List[Tuple[Path, int]]] = {}

    def discover(self):

        measurement_dir = self.base_dir / "measurement"
        if not measurement_dir.is_dir():
            raise ValueError(
                f"No measurement/ directory in {self.base_dir}")

        metadata_paths = {
            key: self.base_dir / subdir
            for key, subdir in METADATA_SUBDIRS.items()
        }

        missing = [k for k, p in metadata_paths.items() if not p.exists()]
        if missing:
            raise ValueError(
                f"Missing metadata in {self.base_dir}: {missing}")

        print(f"Scanning {measurement_dir} ...")

        found = 0
        for meta_file in measurement_dir.rglob("metadata"):
            run_dir = meta_file.parent
            config, version = self._resolve(meta_file, metadata_paths)

            if config is not None and version is not None:
                self.grouped_data.setdefault(config, []).append(
                    (run_dir, version))
                found += 1

        print(f"  Found {found} valid measurements "
              f"across {len(self.grouped_data)} configs")

        self._deduplicate()

    def _resolve(self, metadata_path: Path,
                 metadata_paths: Dict[str, Path]
                 ) -> Tuple[Optional[Config], Optional[int]]:

        meta = self._read_json(metadata_path)
        if not meta:
            return None, None

        try:
            host_id = int(meta["machine_host"])
            install_id = int(meta["platform_installation"])
            gc_id = int(meta["configuration"])
            workload_id = int(meta["benchmark_workload"])
        except (ValueError, TypeError, KeyError):
            return None, None

        if host_id not in ALLOWED_MACHINE_HOSTS:
            return None, None
        if gc_id not in ALLOWED_GC_CONFIGS:
            return None, None

        platform_id, version = self._resolve_platform(
            install_id, metadata_paths)
        if platform_id not in ALLOWED_PLATFORM_TYPES:
            return None, None

        bench_name = self._resolve_benchmark(
            workload_id, metadata_paths)
        if not bench_name:
            return None, None

        config = Config(
            benchmark_type=bench_name,
            machine_host=host_id,
            platform_type=platform_id,
            gc_config=gc_id,
        )
        return config, version

    def _resolve_platform(self, install_id: int,
                          metadata_paths: Dict[str, Path]
                          ) -> Tuple[Optional[int], Optional[int]]:

        data = self._read_json(metadata_paths["platform_installation"])
        record = data.get(str(install_id))
        if not record:
            return None, None
        try:
            return int(record["type"]), int(record["version"])
        except (KeyError, ValueError, TypeError):
            return None, None

    def _resolve_benchmark(self, workload_id: int,
                           metadata_paths: Dict[str, Path]
                           ) -> Optional[str]:

        data = self._read_json(metadata_paths["benchmark_workload"])
        record = data.get(str(workload_id))
        if not record:
            return None
        return record.get("name")

    def _read_json(self, path: Path) -> dict:
        if path in self._json_cache:
            return self._json_cache[path]
        try:
            with path.open('r') as f:
                data = json.load(f)
                self._json_cache[path] = data
                return data
        except Exception:
            return {}

    def _deduplicate(self):

        total = 0

        for config, entries in self.grouped_data.items():

            version_map: Dict[int, List[Tuple[Path, int]]] = {}
            for path, version in entries:
                version_map.setdefault(version, []).append((path, version))

            kept = []
            for _, paths in version_map.items():
                if len(paths) == 1:
                    kept.append(paths[0])
                else:
                    best = max(paths, key=lambda x: _count_rows(x[0]))
                    kept.append(best)

            kept.sort(key=lambda x: x[1])
            self.grouped_data[config] = kept
            total += len(kept)

        print(f"  After deduplication: {total} measurements")


def _count_rows(run_dir: Path) -> int:
    csv_path = _find_csv(run_dir)
    if csv_path is None:
        return 0
    try:
        return sum(1 for _ in open(csv_path)) - 1
    except Exception:
        return 0


def discover_all_runs(base_dir: Path) -> Dict[Config, List[Tuple[Path, int]]]:

    loader = DataLoader(base_dir=base_dir)
    loader.discover()
    return loader.grouped_data