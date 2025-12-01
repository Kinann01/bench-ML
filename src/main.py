#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, List, Optional

from analyzer import Analyzer
from config import Config
from detectors import SteadyStateDetector
from finder import DataFinder
from logger import logger
from pipeline import DatasetPipeline
from plotter import Plotter
from util import DatasetUtil

ROOT = Path(__file__).resolve().parent.parent

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
CONFIG_FILE_NAME = "config.json"

def load_and_validate_configs(config_path: Path) -> List[Config]:
    try:
        with config_path.open('r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON in {config_path}: {e}")

    if not isinstance(raw_data, list):
        raise ValueError(f"Invalid JSON structure in {config_path}. Expected a top-level array (list) of configs.")

    validated_configs : List[Config] = []

    for index, raw_config in enumerate(raw_data):
        try:

            if not isinstance(raw_config, dict):
                raise ValueError("Item is not a JSON object/dictionary.")

            benchmark_type = raw_config["benchmark_type"]
            machine_host = int(raw_config["machine_host"])
            platform_type = int(raw_config["platform_type"])
            gc_config = int(raw_config["gc_config"])

            if machine_host not in ALLOWED_MACHINE_HOSTS:
                raise ValueError(f"MACHINE_TYPE '{machine_host}' is invalid. Must be one of {ALLOWED_MACHINE_HOSTS}.")

            if platform_type not in ALLOWED_PLATFORM_TYPES:
                allowed_ids = list(ALLOWED_PLATFORM_TYPES.keys())
                raise ValueError(f"PLATFORM_TYPE ID '{platform_type}' is invalid. Must be one of {allowed_ids} (JIT only).")

            if gc_config not in ALLOWED_GC_CONFIGS:
                raise ValueError(f"GC_CONFIG ID '{gc_config}' is invalid. Must be one of {ALLOWED_GC_CONFIGS}.")

            config_obj = Config(
                benchmark_type=benchmark_type,
                machine_host=machine_host,
                platform_type=platform_type,
                gc_config=gc_config,
            )
            validated_configs.append(config_obj)

        except KeyError as e:
            raise ValueError(f"Missing required key in config item #{index}: {e}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid configuration in item #{index}: {e}")

    logger.info(f"Successfully loaded and validated {len(validated_configs)} configurations.")

    return validated_configs


def main(config_file: str, base_dir : Optional[Path] = None):

    try:
        config_path = ROOT / Path(config_file)
        configs = load_and_validate_configs(config_path)
        logger.info(f"Running with: {len(configs)} configs from configs.json")

        finder = DataFinder(configs=configs, base_dir=base_dir)
        plotter = Plotter(ROOT / "plots")

        all_paths_found = finder.get_paths()

        if not all_paths_found:
            raise ValueError("No data found")

        analyzer = Analyzer(plotter=plotter)
        detector = SteadyStateDetector(plotter=plotter)
        pipeline = DatasetPipeline(detector=detector, plotter=plotter)

        for config in configs:
            paths = all_paths_found[config]
            versions = finder.get_versions(config)
            merged_dataset = pipeline.process_and_merge(paths=paths, versions=versions)
            df_util = DatasetUtil(df=merged_dataset, versions=versions, config=config)

            if df_util.needs_analysis():
                analyzer.set_context(df_util)
                analyzer.analyze()

            ## report

    except (ValueError, FileNotFoundError) as _:
        pass
    except Exception as _:
        pass
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass

if __name__ == "__main__":
    main(CONFIG_FILE_NAME)
