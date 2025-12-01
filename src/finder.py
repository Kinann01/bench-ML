#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from config import Config
from logger import logger

""""
Reminder:

1) _drop_poor() is important for cases where the SAME benchmark was run under
the same configuration multiple times. We dont need all of them becuase they are
duplicates, we will take the richest.

2) unique machine host is important. If you think we can take a unique machine type
rather than a unique machine host then thats wrong. This is because we can have
different machine hosts pointing to the same machine type but ran under the
same platform type and GC config. i.e. _drop_poor() will have to do more work.
"""

ROOT = Path(__file__).resolve().parent.parent
MIN = -1

_METADATA_FILES: Dict[str, Path] = {
    "benchmark_workload": Path(ROOT / "metadata" /  "benchmark_workload/metadata"),
    "machine_host": Path(ROOT / "metadata" / "machine_host/metadata"),
    "platform_installation": Path(ROOT / "metadata" / "platform_installation/metadata"),
    "platform_type": Path(ROOT / "metadata" / "platform_type/metadata"),
    "version": Path(ROOT / "metadata" / "version/metadata"),
    "repository": Path(ROOT / "metadata" / "repository/metadata"),
}

class DataFinder:
    """
    self.configs: List[Config] -> User provided configurations
    self.base_dir: Path -> starting point for searching data
    self._paths: Dict[Config, Set[Path]] -> Maps a Config to a set of found paths
    self._json_cache: Dict[Path, Any] -> cache of parsed JSON data to avoid re-reading
    self._version_cache: Dict[Config, Dict[Path, str]] -> Per-config version cache
    self.count: int -> total number of data files found across all configs
    """

    def __init__(self, configs: List[Config], *, base_dir: Optional[Path] = None) -> None:
        self.configs = configs
        self.base_dir: Path = ROOT / "measurement" if base_dir is None else base_dir
        logger.info(f"Init DataFinder with base_dir: {self.base_dir} for {len(configs)} configs")
        self._paths: Dict[Config, Set[Path]] = {cfg: set() for cfg in self.configs}
        self._version_cache: Dict[Config, Dict[Path, str]] = {cfg: {} for cfg in self.configs}
        self._json_cache: Dict[Path, Any] = {}
        self.count = 0
        self._scan_disk()

    def get_paths(self) -> Dict[Config, Set[Path]]:
        if not any(self._paths.values()):
            self._scan_disk()
        return self._paths

    def get_versions(self, config : Config) -> Dict[Path, str]:
        return self._version_cache[config]

    def _scan_disk(self):
        logger.info("Starting scan for measurements matching configurations...")

        for metadata_file in self.base_dir.rglob("metadata"):
            root = metadata_file.parent
            config = self._find_matching_config(metadata_file)

            if config:
                self._store_version(root, metadata_file, config)
                self._paths[config].add(root)

        try:
            self._drop_poor()
        except Exception:
            logger.exception("Error in _drop_poor()")

        self.count = sum(len(paths) for paths in self._paths.values())
        logger.info("Scan complete. Found %d matching directories across %d configs.", self.count, len(self.configs))

    def _find_matching_config(self, metadata_path: Path) -> Optional[Config]:

        metadata_file = self._read_json(metadata_path)

        try:
            machine_host = metadata_file["machine_host"]
            platform_installation = metadata_file["platform_installation"]
            configuration = metadata_file["configuration"]
            benchmark_workload = metadata_file["benchmark_workload"]
        except KeyError as e:
            logger.debug(
                "Skipping %s: Missing required metadata key: %s",
                metadata_path.name, e
            )
            return None

        for config in self.configs:
            if (
                self._verify_benchmark_type(benchmark_workload, config)
                and self._verify_machine_host(machine_host, config)
                and self._verify_platform_type(platform_installation, config)
                and self._verify_gc_config(configuration, config)
            ):
                return config

        return None

    def _verify_benchmark_type(self, benchmark_workload: int, target_config: Config) -> bool:

        jfile = self._read_json(_METADATA_FILES["benchmark_workload"])
        try:
            return target_config.benchmark_type in jfile[str(benchmark_workload)]["name"]
        except KeyError:
            return False

    def _verify_machine_host(self, machine_host_value: int, target_config: Config) -> bool:
        try:
            return machine_host_value == target_config.machine_host
        except KeyError:
            return False

    def _verify_platform_type(self, platform_installation: int, target_config: Config) -> bool:
        try:
            pi_json = self._read_json(_METADATA_FILES["platform_installation"])
            pi_record = pi_json[str(platform_installation)]
            platform_type = int(pi_record["type"])
            return platform_type == target_config.platform_type
        except (KeyError, ValueError):
            return False

    def _verify_gc_config(self, configuration_value: int, target_config: Config) -> bool:
        try:
            gc_config = int(configuration_value)
            return gc_config == target_config.gc_config
        except ValueError:
            return False

    def _store_version(self, root: Path, metadata: Path, config: Config) -> None:
        metadata_file = self._read_json(metadata)
        pi_json = self._read_json(_METADATA_FILES["platform_installation"])
        platform_installation = metadata_file["platform_installation"]

        try:
            pi_record = pi_json[str(platform_installation)]
            version = str(pi_record["version"])
        except KeyError:
             logger.warning(f"Could not find version info for {root}")
             return

        if not version:
            raise ValueError(f"Invalid version: {version}")

        if version == self._version_cache[config].get(root):
            return

        self._version_cache[config][root] = version

    def _drop_poor(self) -> None:

        def _count_instances(path: Path) -> int:
            p = path / "default.csv"
            if not p.exists() or not p.is_file():
                return 0
            import pandas as pd
            try:
                df_iter = pd.read_csv(p, chunksize=100_000, dtype=str, low_memory=True)
                return int(sum(len(chunk) for chunk in df_iter))
            except Exception:
                return 0

        for config in self.configs:
            current_version_map = self._version_cache[config]

            versions_to_paths: Dict[str, List[Path]] = {}
            for path, version in current_version_map.items():
                if version not in versions_to_paths:
                    versions_to_paths[version] = []
                versions_to_paths[version].append(path)

            kept_paths: Set[Path] = set()

            for _, paths in versions_to_paths.items():

                if len(paths) == 1:
                    kept_paths.add(paths[0])
                    continue

                # Duplicate, find richest
                best_path = None
                best_count = MIN

                for p in paths:
                    temp_count = _count_instances(p)
                    if temp_count >= best_count:
                        best_count = temp_count
                        best_path = p

                if best_path is not None:
                    kept_paths.add(best_path)

            self._paths[config] = kept_paths
            self._version_cache[config] = {
                p: v for p, v in current_version_map.items() if p in kept_paths
            }

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if path in self._json_cache:
            return self._json_cache[path]

        data: Dict[str, Any] = {}
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Error reading JSON file %s: %s", path, e.__class__.__name__)
        except Exception as e:
            logger.error("Unexpected error reading JSON file %s: %s", path, e)

        self._json_cache[path] = data
        return data

if __name__ == "__main__":
    pass
