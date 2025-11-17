#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from config import Config
from logger import logger

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
    self.config: Config -> User provided configuration
    self.base_dir: Path -> starting point for searching data
    self._paths: Set[Path] -> set of paths to data files
    self._json_cache: Dict[Path, Any] -> cache of parsed JSON data to avoid re-reading
    self._version_cache: Dict[Path, str] -> cache of parsed version data for quick access to repo name
    self.count: int -> number of data files found
    """

    def __init__(self, config: Config, *, base_dir: Optional[Path] = None) -> None:
        self.config = config
        self.base_dir: Path = ROOT / "measurement" if base_dir is None else base_dir
        logger.info(f"Init DataFinder with base_dir: {self.base_dir}")
        self._paths: Set[Path] = set()
        self._json_cache: Dict[Path, Any] = {}
        self._version_cache : Dict[Path, str]= {}
        self.count = 0
        self._scan_disk()

    def get_paths(self) -> Set[Path]:
        if not self._paths:
            self._scan_disk()
        return self._paths # TODO: return a RO proxy - we dont want to call .copy() to avoid new allocation

    def get_versions(self) -> Dict[Path, str]:
        if not self._version_cache:
            self._scan_disk()
        return self._version_cache # TODO: return a RO proxy - we dont want to call .copy() to avoid new allocation

    def _scan_disk(self):
        logger.info("Starting scan for measurements matching configuration...")
        for metadata_file in self.base_dir.rglob("metadata"):
            root = metadata_file.parent
            if self._is_useful_metadata(metadata_file):
                self._store_version(root, metadata_file)
                self._paths.add(root)

        try:
            """ See self._drop_poor()"""
            self._drop_poor()
        except Exception:
            logger.exception("Error in _drop_poor()")
        logger.info("Scan complete. Found %d matching directories.", self.count)

    def _is_useful_metadata(self, metadata_path: Path) -> bool:
        """
        Check if the metadata file contains all required fields and matches
        the configured filter values.
        """
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
            return False

        return (
            self._verify_benchmark_type(benchmark_workload)
            and self._verify_machine_host(machine_host)
            and self._verify_platform_type(platform_installation)
            and self._verify_gc_config(configuration)
        )

    def _verify_benchmark_type(self, benchmark_workload: int) -> bool:
        """Verify the measurement's benchmark_type name matches the config."""
        jfile = self._read_json(_METADATA_FILES["benchmark_workload"])
        try:
            return self.config.benchmark_type in jfile[str(benchmark_workload)]["name"]
        except KeyError:
            return False

    def _verify_machine_host(self, machine_host_value: int) -> bool:
        try:
            return machine_host_value == self.config.machine_host
        except KeyError:
            return False

    def _verify_platform_type(self, platform_installation: int ) -> bool:
        """
        Verify the PI links to the single Platform Type ID specified in the config.
        """
        try:
            pi_json = self._read_json(_METADATA_FILES["platform_installation"])
            pi_record = pi_json[str(platform_installation)]
            platform_type = int(pi_record["type"])
            return platform_type == self.config.platform_type
        except KeyError:
            return False
        except ValueError:
            return False

    def _verify_gc_config(self, configuration_value: int) -> bool:
        try:
            gc_config = int(configuration_value)
            return gc_config == self.config.gc_config
        except ValueError:
            return False

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

    def _drop_poor(self) -> None:

        """
        De-duplicate self._version_cache (path -> version) so each version appears only once.
        When multiple paths have the same version, keep the path whose <path>/default.csv has
        the largest number of instances (rows). Mutates self._version_cache and self._paths in-place
        """

        def _count_instances(path: Path) -> int:
            p = path / "default.csv"

            if not p.exists() or not p.is_file():
                return 0

            import pandas as pd
            count = 0
            df = pd.read_csv(p, chunksize=100_000, dtype=str, low_memory=True)
            count = int(sum(len(chunk) for chunk in df))
            return count

        versions_to_paths: Dict[str, List[Path]] = {}

        for path, version in self._version_cache.items():
            if version not in versions_to_paths:
                versions_to_paths[version] = []
            versions_to_paths[version].append(path)

        kept_paths: Set[Path] = set()
        count: int = 0

        for _, paths in versions_to_paths.items():
            if len(paths) == 1:
                kept_paths.add(paths[0])
                count += 1
                continue

            """ Not unique, lets keep the richest """
            best_path = None
            best_count = MIN
            for p in paths:
                temp_count = _count_instances(p)
                if (temp_count >= best_count):
                    best_count = temp_count
                    best_path = p

            if best_path is not None:
                count += 1
                kept_paths.add(best_path)

        # mutate cache, paths and count
        self._version_cache = {p: v for p, v in self._version_cache.items() if p in kept_paths}
        self._paths = kept_paths
        self.count = count

        for path in kept_paths:
            logger.info(f"Keeping path: {path}")

    def _store_version(self, root: Path, metadata: Path) -> None:
        metadata_file = self._read_json(metadata)
        pi_json = self._read_json(_METADATA_FILES["platform_installation"])
        platform_installation = metadata_file["platform_installation"]
        pi_record = pi_json[str(platform_installation)]
        version = str(pi_record["version"])

        if not version:
            raise ValueError(f"Invalid version: {version}")

        if version == self._version_cache.get(root):
            return

        self._version_cache[root] = version

if __name__ == "__main__":
    pass
