#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, Set

import pandas as pd

from detectors import SteadyStateDetector
from logger import logger
from plotter import Plotter


class DatasetPipeline:
    """
    Responsible for loading files, applying the detector,
    slicing the data, and merging into one dataset.
    """
    def __init__(self, detector: SteadyStateDetector, plotter: Plotter):
        self.detector = detector
        self.plotter = plotter

    def process_and_merge(self, paths: Set[Path], versions : Dict[Path, str]) -> pd.DataFrame:

        processed_frames = []

        logger.info("Processing and merging datasets...")
        logger.info("------------------------------")

        for path in paths:
            csv_path = path / "default.csv"

            if not csv_path.exists():
                logger.warning(f"File not found: {csv_path}")
                continue

            try:
                raw_df = pd.DataFrame(pd.read_csv(csv_path))
                logger.info("Processing file: %s", csv_path)
                self.plotter.set_context(csv_path)
                cutoff_idx = self.detector.detect_cutoff_index(raw_df)
                logger.info("Finished processing file: %s", csv_path)

                if cutoff_idx == 0:
                    logger.warning(f"No cutoff index detected for {csv_path}")
                    continue

                logger.info("Detected cutoff index: %d", cutoff_idx)
                logger.info("------------------------------")
                steady_df = raw_df.iloc[cutoff_idx:].copy()
                steady_df['version'] = versions[path]
                processed_frames.append(steady_df)

            except Exception as e:
                logger.error(f"Failed to process {csv_path}: {e}")

        if not processed_frames:
            logger.info("No files processed")
            return pd.DataFrame()

        merged_df = pd.concat(processed_frames, ignore_index=True)
        return merged_df


if __name__ == "__main__":
    pass
