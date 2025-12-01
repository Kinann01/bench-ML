#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from config import Config
from logger import logger


class SpikinessLevel(Enum):
    NOT_SPIKY = 0
    MID_SPIKE = 1
    HIGH_SPIKY = 2

class DatasetUtil:

    """
    [,0.02] is considered low
    [0.02, 0.05] is considered mid
    [0.05,] is considered high
    """
    THRESHOLD_MID = 0.02
    THRESHOLD_HIGH = 0.05

    def __init__(self, df: pd.DataFrame, versions: Dict[Path, str], config: Config):
        self.df= df
        self.versions = versions
        self.config = config
        self._spike_levels : Dict[Path, float] = {}

    def _calculate_metric(self, series: pd.Series) -> float:
        standard_deviation = series.std()
        mean_val = series.mean()
        return standard_deviation / mean_val

    def _get_category(self, score: float) -> SpikinessLevel:
        if score < self.THRESHOLD_MID:
            return SpikinessLevel.NOT_SPIKY
        elif score < self.THRESHOLD_HIGH:
            return SpikinessLevel.MID_SPIKE
        else:
            return SpikinessLevel.HIGH_SPIKY

    def _smoothen(self, series: pd.Series, window_size: int, percentile: float = 95.0) -> pd.Series:
        smoothed = series.copy()
        n = len(series)

        lower_quantile = (100 - percentile) / 200
        upper_quantile = 1 - lower_quantile

        for i in range(0, n, window_size):
            temp_min = min(i + window_size, n)
            subset_slice = slice(i, temp_min)
            subset = series.iloc[subset_slice]

            median = subset.median()
            lower_bound = subset.quantile(lower_quantile)
            upper_bound = subset.quantile(upper_quantile)
            outlier_mask = (subset < lower_bound) | (subset > upper_bound)
            smoothed.loc[subset[outlier_mask].index] = median

        return smoothed

    def _build_spikeness_levels(self):

        target_col = 'iteration_time_ns'

        for path, version_str in self.versions.items():

            subset = pd.DataFrame(self.df[self.df['version'] == version_str])

            if subset.empty:
                continue

            n = int(np.clip(len(subset) * 0.10, 5, 200))
            val = self._calculate_metric(self._smoothen(pd.Series(subset[target_col]), window_size=n))
            self._spike_levels[path] = val
            logger.debug(f"Calculated spike level for {path}: {val}")

    def needs_analysis(self) -> bool:

        categories = [self._get_category(score) for score in self._spike_levels.values()]

        if not categories:
            return True

        has_clean = SpikinessLevel.NOT_SPIKY in categories
        has_dirty = (SpikinessLevel.MID_SPIKE in categories) or \
                    (SpikinessLevel.HIGH_SPIKY in categories)

        return False if has_clean and has_dirty else True

    def get_spikeness(self, path: Path) -> float:
        return self._spike_levels.get(path, 0.0)

    def calculate_spikeness(self, path: Path) -> float:
        dataset = pd.DataFrame(pd.read_csv(path / 'default.csv'))
        series = pd.Series(dataset['iteration_time_ns'])
        return self._calculate_metric(series)

    def get_data(self) -> pd.DataFrame:
        return self.df

    def get_config(self) -> Config:
        return self.config

    def get_versions(self) -> Dict[Path, str]:
        return self.versions
