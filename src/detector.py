#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Optional

from pipeline_config import DEFAULTS
from constants import COMPILATION_TIME_COLS

_ssd_defaults = DEFAULTS["ssd"]
WINDOW_FRACTION = _ssd_defaults["window_fraction"]
WINDOW_MIN = _ssd_defaults["window_min"]
WINDOW_MAX = _ssd_defaults["window_max"]
THRESHOLD_FRACTION = _ssd_defaults["threshold_fraction"]


def _resolve_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def configure(ssd_params: dict):
    global WINDOW_FRACTION, WINDOW_MIN, WINDOW_MAX, THRESHOLD_FRACTION
    WINDOW_FRACTION = ssd_params.get("window_fraction", WINDOW_FRACTION)
    WINDOW_MIN = ssd_params.get("window_min", WINDOW_MIN)
    WINDOW_MAX = ssd_params.get("window_max", WINDOW_MAX)
    THRESHOLD_FRACTION = ssd_params.get("threshold_fraction", THRESHOLD_FRACTION)


class SteadyStateDetector:

    def detect_cutoff_index(self, df: pd.DataFrame) -> int:

        if df.empty:
            return 0

        cas_col = _resolve_col(df, COMPILATION_TIME_COLS)
        if cas_col is None:
            return 0

        n = len(df)
        window_size = int(np.clip(n * WINDOW_FRACTION, WINDOW_MIN, WINDOW_MAX))
        threshold = float(np.max(df[cas_col]) * THRESHOLD_FRACTION)

        return self._cas_analysis(df[cas_col], window_size, threshold)

    def _cas_analysis(self, cas_series: pd.Series,
                      window_size: int, threshold: float) -> int:

        n = cas_series.shape[0]
        if window_size <= 0 or n < window_size:
            return 0

        kernel = np.ones(window_size, dtype=np.float32)
        rolling_mean = np.convolve(cas_series, kernel, mode='valid') / window_size

        candidates = np.where(rolling_mean <= threshold)[0]
        return int(candidates[0]) if candidates.size > 0 else 0


def main():
    print("This is a module, not meant to be run directly.")