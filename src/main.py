#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent

from config import Config
from detectors import SteadyStateDetector
from finder import DataFinder
from logger import logger
from pipeline import DatasetPipeline
from plotter import Plotter

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
ALLOWED_MACHINE_HOSTS: List[int] = [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]
CONFIG_FILE_NAME = "config.json"

def load_and_validate_config(config_path: Path) -> Config:

    try:
        with config_path.open('r') as f:
            raw_config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON in {config_path}: {e}")

    try:
        benchmark_type = raw_config["benchmark_type"]
        machine_host = int(raw_config["machine_host"])
        platform_type = int(raw_config["platform_type"])
        gc_config = int(raw_config["gc_config"])

    except KeyError as e:
        raise ValueError(f"Missing required key in config.json: {e}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid type or value in config.json: {e}")

    if machine_host not in ALLOWED_MACHINE_HOSTS:
        raise ValueError(f"MACHINE_TYPE '{machine_host}' is invalid. Must be one of {ALLOWED_MACHINE_HOSTS}.")

    if platform_type not in ALLOWED_PLATFORM_TYPES:
        allowed_ids = list(ALLOWED_PLATFORM_TYPES.keys())
        raise ValueError(f"PLATFORM_TYPE ID '{platform_type}' is invalid. Must be one of {allowed_ids} (JIT only).")

    if gc_config not in ALLOWED_GC_CONFIGS:
        raise ValueError(f"GC_CONFIG ID '{gc_config}' is invalid. Must be one of {ALLOWED_GC_CONFIGS}.")

    logger.info("Configuration successfully loaded and validated.")

    return Config(
        benchmark_type=benchmark_type,
        machine_host=machine_host,
        platform_type=platform_type,
        gc_config=gc_config,
    )

def main(config_file: str, base_dir : Optional[Path] = None):

    try:
        config_path = ROOT / Path(config_file)
        config = load_and_validate_config(config_path)
        logger.info(f"Running with: {config}")
        finder = DataFinder(config, base_dir=base_dir)

        plotter = Plotter(ROOT / "plots") # TODO: Should it be fixed or should the user decide?
        detector = SteadyStateDetector(plotter=plotter)
        versions = finder.get_versions()
        pipeline = DatasetPipeline(detector=detector, versions=versions, plotter=plotter)

        found_paths = finder.get_paths()

        if not found_paths:
            raise ValueError("No data found")

        logger.info("Processing and merging datasets...")
        merged_dataset = pipeline.process_and_merge(found_paths)

        """TODO: error handling + pass the merged datasets to the analyzer """

    except (ValueError, FileNotFoundError) as _:
        pass
    except Exception as _:
        pass

if __name__ == "__main__":
    main(CONFIG_FILE_NAME)
