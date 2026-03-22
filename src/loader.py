
#! /usr/bin/env python3
"""
Multi-year DataLoader with per-year metadata resolution.

Expected directory structure:
    base_dir/
        2016/
            measurement/       <- measurement data (singular)
            metadata/          <- metadata for this year
                benchmark_workload/metadata
                machine_host/metadata
                platform_installation/metadata
                platform_type/metadata
        2017/
            measurement/
            metadata/
        ...
        2022/
            measurement/
            metadata/

Each year directory has its own metadata files.
When processing measurements from e.g. 2016/measurement/,
metadata is resolved from 2016/metadata/.
"""

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from detector import SteadyStateDetector
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent

# Padding sentinel: -999 is far outside any log-IQR normalized range
PAD_VALUE = -999.0

# Flexible naming: try each in order
CSV_FILENAMES = ['default.csv', 'default.csv.one-per-rep.csv']
ITERATION_TIME_COLS = ['iteration_time_ns', 'pol_dd_0_iteration_time_ns']


def _find_csv(run_dir: Path) -> Optional[Path]:
    """Return the first existing CSV file from the known name variants, or None."""
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

# --- Constants from data.md ---
ALLOWED_PLATFORM_TYPES: Dict[int, str] = {
    12: "graal-ee-master-jdk-11",
    14: "graal-ee-release-jdk-11",
    15: "graal-ee-release-jdk-8",
    16: "graal-ce-master-jdk-11",
    26: "graal-ce-master-jdk-17",
    27: "graal-ee-master-jdk-17",
    28: "graal-ee-release-jdk-17",
}

ALLOWED_GC_CONFIGS: List[int] = [34, 35, 43]
ALLOWED_MACHINE_HOSTS: List[int] = [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 43, 63]

# Matches year directories: 2015 through 2022
YEAR_DIR_REGEX = re.compile(r"^(201[5-9]|202[0-2])$")

# Metadata subdirectory names (relative to each year dir)
METADATA_SUBDIRS = {
    "benchmark_workload": "metadata/benchmark_workload/metadata",
    "machine_host": "metadata/machine_host/metadata",
    "platform_installation": "metadata/platform_installation/metadata",
    "platform_type": "metadata/platform_type/metadata",
}


@dataclass(frozen=True)
class Config:
    benchmark_type: str
    machine_host: int
    platform_type: int
    gc_config: int


class MultiYearDataLoader:
    """
    DataLoader that discovers year directories (2016/, 2017/, etc.)
    and resolves metadata per-year.

    Each year directory is expected to have:
        measurement/    - contains the benchmark run data
        metadata/       - contains the JSON metadata files
    """

    def __init__(self, base_dir: Path):
        if base_dir is None:
            raise ValueError("base_dir cannot be None")

        self.base_dir = base_dir
        self._json_cache: Dict[Path, Any] = {}
        self.grouped_data: Dict[Config, List[Tuple[Path, str]]] = {}
        self.detector = SteadyStateDetector()
        self.config_scalers: Dict[Config, Tuple[float, float]] = {}  # {config: (median_log, iqr_log)}
        self.max_len = 0

    def _get_metadata_paths(self, year_dir: Path) -> Dict[str, Path]:
        """
        Build metadata paths relative to a specific year directory.

        Args:
            year_dir: e.g. base_dir/2016/

        Returns:
            Dict like {"benchmark_workload": base_dir/2016/metadata/benchmark_workload/metadata, ...}
        """
        return {key: year_dir / subdir for key, subdir in METADATA_SUBDIRS.items()}

    def discover_and_validate(self):
        logger.info(f"Scanning for year directories in {self.base_dir}")

        # Discover all year directories
        year_dirs = []
        for child in sorted(self.base_dir.iterdir()):
            if child.is_dir() and YEAR_DIR_REGEX.match(child.name):
                year_dirs.append(child)

        if not year_dirs:
            raise ValueError(f"No valid year directories (2015-2022) found in {self.base_dir}")

        logger.info(f"Found {len(year_dirs)} year directories: {[d.name for d in year_dirs]}")

        found_count = 0

        for year_dir in year_dirs:
            measurement_dir = year_dir / "measurement"
            if not measurement_dir.is_dir():
                logger.warning(f"Skipping {year_dir.name}: no measurement/ subdirectory")
                continue

            metadata_paths = self._get_metadata_paths(year_dir)

            # Check that metadata exists
            missing_metadata = [k for k, p in metadata_paths.items() if not p.exists()]
            if missing_metadata:
                logger.warning(f"Skipping {year_dir.name}: missing metadata for {missing_metadata}")
                continue

            logger.info(f"Processing {year_dir.name}/measurement/ with {year_dir.name}/metadata/")

            for meta_file in measurement_dir.rglob("metadata"):
                run_dir = meta_file.parent
                config, version = self._validate_and_extract_info(meta_file, metadata_paths)

                # Confirm version as well. Each run must have a resolvable version
                if config and version is not None:
                    self.grouped_data.setdefault(config, []).append(
                        (run_dir, version)
                    )
                    found_count += 1

        logger.info(f"Initial scan complete. Found {found_count} valid measurement directories across {len(self.grouped_data)} unique configs.")
        self._deduplicate_versions()

    def _validate_and_extract_info(self, metadata_path: Path, metadata_paths: Dict[str, Path]) -> Tuple[Optional[Config], Optional[str]]:
        try:
            meta = self._read_json(metadata_path)
        except Exception:
            return None, None

        try:
            m_host_id = int(meta["machine_host"])
            p_install_id = int(meta["platform_installation"])
            gc_config_id = int(meta["configuration"])
            bench_workload_id = int(meta["benchmark_workload"])
        except (ValueError, TypeError, KeyError):
            return None, None

        if m_host_id not in ALLOWED_MACHINE_HOSTS: return None, None
        if gc_config_id not in ALLOWED_GC_CONFIGS: return None, None

        p_type_id, version = self._resolve_platform_info(p_install_id, metadata_paths)
        if p_type_id not in ALLOWED_PLATFORM_TYPES: return None, None

        bench_name = self._resolve_benchmark_name(bench_workload_id, metadata_paths)
        if not bench_name: return None, None

        config = Config(
            benchmark_type=bench_name,
            machine_host=m_host_id,
            platform_type=p_type_id,
            gc_config=gc_config_id
        )
        return config, version

    def _resolve_platform_info(self, p_install_id: int, metadata_paths: Dict[str, Path]) -> Tuple[Optional[int], Optional[str]]:
        path = metadata_paths["platform_installation"]
        data = self._read_json(path)
        if not data: return None, None
        record = data.get(str(p_install_id))
        if not record: return None, None
        try:
            return int(record.get("type")), str(record.get("version"))
        except: return None, None

    def _resolve_benchmark_name(self, bench_workload_id: int, metadata_paths: Dict[str, Path]) -> Optional[str]:
        path = metadata_paths["benchmark_workload"]
        data = self._read_json(path)
        if not data: return None
        record = data.get(str(bench_workload_id))
        if not record: return None
        return record.get("name")

    def _read_json(self, path: Path) -> Dict:
        if path in self._json_cache: return self._json_cache[path]
        try:
            with path.open('r') as f:
                data = json.load(f)
                self._json_cache[path] = data
                return data
        except: return {}

    def _deduplicate_versions(self):
        """
        Deduplicates datasets: valid configs may have multiple runs for the same commit.
        Keeps only the richest (most rows) dataset per version per config.
        """
        logger.info("Deduplicating versions...")
        total_kept = 0
        MIN = -1

        def _count_instances(path: Path) -> int:
            csv_path = _find_csv(path)
            if csv_path is None:
                return 0
            try:
                df_iter = pd.read_csv(csv_path, chunksize=100_000, dtype=str, low_memory=True)
                return int(sum(len(chunk) for chunk in df_iter))
            except Exception:
                return 0

        for config, entries in self.grouped_data.items():
            version_map: Dict[str, List[Path]] = {}

            for path, version in entries:
                if version not in version_map:
                    version_map[version] = []
                version_map[version].append(path)

            kept_paths = []

            for ver, paths in version_map.items():
                if len(paths) == 1:
                    kept_paths.append((paths[0], ver))
                    continue

                best_path = None
                best_count = MIN

                for p in paths:
                    temp_count = _count_instances(p)
                    if temp_count >= best_count:
                        best_count = temp_count
                        best_path = p

                if best_path is not None:
                    kept_paths.append((best_path, ver))

            kept_paths.sort(key=lambda x: x[1])
            self.grouped_data[config] = kept_paths
            total_kept += len(kept_paths)

        logger.info(f"Deduplication complete. Retained {total_kept} valid datasets.")

    def load_and_preprocess_data(self, target_max_len: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Loads all valid data, applies steady-state cutoff, padding, and
        per-config Z-score normalization.

        Normalization: Log + Robust Z-Score (IQR)
        ==========================================
        Benchmark iteration times have multiplicative noise (GC pauses double
        or triple the time). Log transform converts this to additive noise.
        Median/IQR scaling is robust to remaining outliers.

        Pipeline per config:
            1. log(x)  — converts multiplicative noise to additive
            2. (log(x) - median) / IQR — robust centering and scaling

        This makes all configs comparable (centered at 0, unit spread) while
        being immune to extreme GC spikes that distort mean/std.

        Returns:
            X (np.ndarray): Shape (N, max_len, 1)
            max_len (int): Padding length used
        """
        # Phase 1: Load raw series per config
        config_series: Dict[Config, List[np.ndarray]] = {}
        raw_lengths = []

        total_files = sum(len(entries) for entries in self.grouped_data.values())
        logger.info(f"Loading {total_files} valid files across {len(self.grouped_data)} configs...")

        for config, entries in self.grouped_data.items():
            series_list = []
            for path, _ in entries:
                csv_path = _find_csv(path)
                if csv_path is None:
                    continue
                try:
                    df = pd.read_csv(csv_path)
                    iter_col = _resolve_iter_col(df)
                    if df.empty or iter_col is None:
                        continue
                    cutoff_idx = self.detector.detect_cutoff_index(df)
                    if cutoff_idx == 0:
                        continue
                    series = df[iter_col].iloc[cutoff_idx:].values.astype(float)
                    series_list.append(series)
                    raw_lengths.append(len(series))
                except Exception as e:
                    logger.error(f"Failed to process file {csv_path}: {e}")

            if series_list:
                config_series[config] = series_list

        if not config_series:
            raise ValueError("No valid series found.")

        # Phase 2: Compute per-config log + IQR statistics
        logger.info(f"Computing per-config log+IQR normalization for {len(config_series)} configs...")
        self.config_scalers = {}

        for config, series_list in config_series.items():
            concat = np.concatenate(series_list)
            concat_log = np.log(concat)
            median = float(np.median(concat_log))
            q25, q75 = np.percentile(concat_log, [25, 75])
            iqr = float(q75 - q25)
            if iqr < 1e-10:  # Guard against near-zero IQR (constant series)
                iqr = 1.0
            self.config_scalers[config] = (median, iqr)

        # Phase 3: Determine max_len
        if target_max_len:
            self.max_len = target_max_len
        else:
            self.max_len = max(raw_lengths)

        logger.info(f"Max sequence length set to: {self.max_len}")

        # Phase 4: Log-transform, normalize, truncate, pad
        processed_data = []

        for config, series_list in config_series.items():
            median, iqr = self.config_scalers[config]
            for s in series_list:
                s_norm = (np.log(s) - median) / iqr  # log + IQR normalization

                if len(s_norm) > self.max_len:
                    s_norm = s_norm[:self.max_len]

                padded = np.full(self.max_len, PAD_VALUE)
                padded[:len(s_norm)] = s_norm
                processed_data.append(padded)

        X = np.array(processed_data)
        X = np.expand_dims(X, axis=-1)

        logger.info(f"Data loading complete. Shape: {X.shape}")
        logger.info(f"Per-config scalers computed for {len(self.config_scalers)} configs")

        return X, self.max_len

    def save_scalers(self, path: str = "config_scalers.pkl"):
        """Save per-config (median_log, iqr_log) scalers for inference."""
        with open(path, 'wb') as f:
            pickle.dump(self.config_scalers, f)
        logger.info(f"Saved {len(self.config_scalers)} config scalers to {path}")

    @staticmethod
    def load_scalers(path: str = "config_scalers.pkl") -> Dict:
        """Load previously saved per-config scalers."""
        with open(path, 'rb') as f:
            scalers = pickle.load(f)
        logger.info(f"Loaded {len(scalers)} config scalers from {path}")
        return scalers


def discover_all_runs(base_dir: Path) -> Dict[Config, List[Tuple[Path, str]]]:
    """
    Scan all year directories under base_dir, validate metadata,
    deduplicate, and return Config -> [(run_dir, version)] mapping.

    Thin wrapper around MultiYearDataLoader for use by build_index.py.
    """
    loader = MultiYearDataLoader(base_dir=base_dir)
    loader.discover_and_validate()
    return loader.grouped_data


def main():
    loader = MultiYearDataLoader(base_dir=ROOT)
    loader.discover_and_validate()
    X, max_len = loader.load_and_preprocess_data()
    np.save("training_data_all_years.npy", X)
    loader.save_scalers("config_scalers.pkl")
    print(f"Saved training_data_all_years.npy — shape: {X.shape}, max_len: {max_len}")
    print(f"Saved config_scalers.pkl — {len(loader.config_scalers)} configs")


if __name__ == "__main__":
    main()
