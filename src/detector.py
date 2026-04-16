#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Optional

COMPILATION_TIME_COLS = ['compilation_time_ms', 'pol_dd_0_compilation_time_ms']

WINDOW_FRACTION = 0.10
WINDOW_MIN = 5
WINDOW_MAX = 50
THRESHOLD_FRACTION = 0.009


def _resolve_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


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
