#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from logger import logger
from plotter import Plotter


@dataclass
class SteadyStateParams:
    window_size: int = 0 # Window size used for CAS analysis and for index comparison
    cas_threshold: float = 0.0 # compilation_time_ns activity
    smoothen_window_size: int = 0 # Window size used for smoothening
    kernel_size: int = 0 # Kernel size used in KB-KSSD
    comparison_window_size: int = 0 # Window size used for comparison

class SteadyStateDetector:

    def __init__(self, plotter : Optional[Plotter] = None):
        self.plotter = plotter
        self.params = SteadyStateParams()

    def _perform_cas_analysis(self, cas_series: pd.Series) -> int:

        if self.params.window_size <= 0:
            logger.error("_perform_cas_analysis: Window size must be greater than 0")
            return -1

        n = cas_series.shape[0]
        if n < self.params.window_size:
            logger.error("_perform_cas_analysis: Data size must be greater than window size")
            return -1

        kernel = np.ones(self.params.window_size, dtype=np.float32)
        rolling_sums = np.convolve(cas_series, kernel, mode='valid')
        rolling_mean = rolling_sums / self.params.window_size
        candidates = np.where(rolling_mean <= self.params.cas_threshold)[0]
        return int(candidates[0]) if candidates.size != 0 else -1

    def _perform_pos_analysis(self, iteration_time_series: pd.Series, roi_start_index: int) -> int:

        pos_roi = iteration_time_series.iloc[slice(roi_start_index, None)]
        smoothed_roi = self._smoothen(pos_roi)

        half_k = self.params.kernel_size // 2
        kernel = np.concatenate([np.ones(half_k), -1 * np.ones(half_k)])
        convolved = np.convolve(smoothed_roi, kernel, mode='valid')
        absolute_index = roi_start_index + (np.argmax(convolved) + half_k)
        return int(absolute_index) if absolute_index < len(iteration_time_series) else -1

    def detect_cutoff_index(self, df: pd.DataFrame) -> int:

        """
        Detects the steady state index for the given DataFrame.

        We combine two algorithms:
            1. Heuristic-based detection in compilation activity; compilation_time_ms
            2. Part of the KB-KSSD algorithm used on iteration_time_ns

        From KB-KSSD:
            1. We smoothen the data using rolling median
            2. Discerete convolution using a kernel of the shape [...,1 , 1, -1, -1, ...]

        With CAS analysis, we find the first drop, which could potentially be the best indicator
        of the stready state index. After that, we run the KB-KSSD algorithm on the remaining data.
        Once KB-KSSD is complete, we have to ensure that the value returned by KB-KSSD is better
        than what we found with CAS analysis.

        Major support is missing for dataFrames with no compilation activity. In this case, we have to fully
        rely on KB-KSSD algorithm but that would require running a kernel of size(dataFrame.rows()) to pick
        the first drop and we do not have support for that.
        """

        if df.empty:
            logger.warning("detect_cutoff_index: DataFrame is empty, returning cutoff 0. Will be skipped")
            return 0

        self._calculate_dynamic_params(df)

        cas_series = df.get('compilation_time_ms', None)
        pos_series = df.get('iteration_time_ns', None)
        cas_signal, pos_signal, cas_offset, pos_offset = -1, -1, -1, -1
        n = len(df)

        if pos_series is None:
            logger.warning("detect_cutoff_index: DataFrame has no iteration_time_ns column. Will be skipped")
            return 0

        if cas_series is None:
            logger.warning("detect_cutoff_index: no compilation activity is available. Cannot rely on POS analysis.")
            logger.warning("Returning cutoff 0, will be skipped")
            return 0

        cas_series = cas_series.copy()
        cas_signal = self._perform_cas_analysis(cas_series)

        logger.debug(f"cas_signal: {cas_signal}")
        start_index_for_pos = cas_signal
        cas_offset = cas_signal

        pos_series = pos_series.copy()
        pos_offset = self._perform_pos_analysis(pos_series, roi_start_index=start_index_for_pos)
        logger.debug(f"pos_signal: {pos_signal}")

        """
        Check and verify whats the best we have so far.
        If CAS analysis failed, we fully rely on POS
        analysis results. Else we compare which is better
        """

        assert pos_offset >= 0

        if cas_offset <= 0:
            if self.plotter:
                self.plotter.log_step(pos_series, "SSD_OFFSET", cas_offset, pos_offset, pos_offset)
            return pos_offset

        verified_index = self._verify_index_by_median_diff(
            series=pos_series,
            new_index=pos_offset,
            original_index=cas_offset,
            window_size=self.params.comparison_window_size
        )

        if self.plotter:
            self.plotter.log_step(pos_series, "SSD_OFFSET", cas_offset, pos_offset, verified_index)

        """
        We dont want our cutoff index to be in the last
        30% of the series. Not a reasonable choice so we
        fall back to randomness and pick something in the
        10-50% of the series.
        """

        last_percent_threshold_idx = int(n * 0.70)
        if verified_index >= last_percent_threshold_idx:
            logger.warning(f"Detected cutoff index {verified_index} is in the last ~30% of the series (total length: {n}). "
                            "Selecting a random index from the first 10-50% as an alternative.")

            min_random_idx = int(n * 0.10)
            max_random_idx = int(n * 0.50)
            new_chosen_index = random.randint(min_random_idx, max_random_idx)
            verified_index = new_chosen_index
            logger.debug(f"New cutoff index selected randomly: {verified_index}")

        return verified_index

    def _get_median_diff(self, series: pd.Series, index: int, window_size: int) -> float:

            left_window = series.iloc[slice(index - window_size, index)]
            right_window = series.iloc[slice(index, index + window_size)]

            if left_window.empty or right_window.empty:
                return 0.0

            left_median = left_window.median()
            right_median = right_window.median()

            diff = left_median - right_median
            return diff if diff > 0 else 0.0

    def _verify_index_by_median_diff(self, series: pd.Series, new_index: int, original_index: int, window_size: int) -> int:

        new_index_diff = self._get_median_diff(series, new_index, window_size)
        original_index_diff = self._get_median_diff(series, original_index, window_size)

        if new_index_diff > original_index_diff:
            return new_index

        return original_index

    def _calculate_dynamic_params(self, df: pd.DataFrame) -> None:

        n = len(df)
        if n == 0:
            return

        self.params.window_size = int(np.clip(n * 0.10, 5, 50))

        compilation_activity = df.get('compilation_time_ms')

        if compilation_activity is None:
            self.params.cas_threshold = 0.0
        else:
            peak_activity = np.max(compilation_activity)
            self.params.cas_threshold = float(peak_activity * 0.009)

        self.params.smoothen_window_size = int(np.clip(n * 0.10, 10, 200))

        raw_kernel = int(np.clip(n * 0.15, 2, 16))

        if raw_kernel % 2 != 0:
            raw_kernel -= 1

        self.params.kernel_size = max(2, raw_kernel)

        self.params.comparison_window_size = int(np.clip(n * 0.10, 5, 100))
        logger.info(f"Dynamic Params for N={n}: {self.params}")

    def _smoothen(self, series: pd.Series, percentile: float = 95.0) -> pd.Series:
        smoothed = series.copy()
        n = len(series)

        lower_quantile = (100 - percentile) / 200
        upper_quantile = 1 - lower_quantile

        for i in range(0, n, self.params.smoothen_window_size):
            temp_min = min(i + self.params.smoothen_window_size, n)
            subset_slice = slice(i, temp_min)
            subset = series.iloc[subset_slice]

            if subset.empty:
                continue

            median = subset.median()
            lower_bound = subset.quantile(lower_quantile)
            upper_bound = subset.quantile(upper_quantile)
            outlier_mask = (subset < lower_bound) | (subset > upper_bound)
            smoothed.loc[subset[outlier_mask].index] = median

        return smoothed

if __name__ == "__main__":
    pass
