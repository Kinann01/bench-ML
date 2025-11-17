#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np
import pandas as pd

from logger import logger
from plotter import Plotter

DUMMY = 0

"""
 Major TODOs in detectors.py:
    Parameters. See below
    Verify typing in the functions.
    Add logging in POS analysis
    Implement SSD with full reliance on KB-KSSD in cases where CAS is not available as such datasets exist.
    signal.fftconvolve() should be faster than np.convolve(). study that and use it
    detect_cutoff_index() needs re-digesting to health-check values returned by the signal analysis funcs()
    _verify_index_by_median_diff() needs re-digesting to deal with index our of bound issues on the edges.
    _perform_pos_analysis() absolute index could be out of bounds?
    plotter.log_step() call it where needed

 Regarding constant parameters:

    We will have to automatically determine these parameters based on data specifics.

    cas_window_size
    cas_threshold
    smoothen_window_size
    smoothen_percentile
    kb_kssd_kernel_size
    verify_window_size

"""

class SteadyStateDetector:

    def __init__(self, window_size : int = 5, threshold : float = 0.05, plotter : Optional[Plotter] = None):
        self.window_size = window_size
        self.threshold = threshold
        self.plotter = plotter

    def _perform_cas_analysis(self, cas_series: pd.Series,
                                    window_size: int,
                                    threshold: float) -> int:

        """
        First quiet window based on CAS - Compiler Activity Signal Analysis.

        returns the index (row number) where the first quiet window begins.
        return -1 if no quiet window is found.
        """

        if window_size <= 0:
            logger.warning("_perform_cas_analysis: Window size must be greater than 0")
            return -1

        column = np.asarray(cas_series.fillna(0.0), dtype=np.float32)
        n = len(column)
        if n < window_size:
            logger.warning("_perform_cas_analysis: Data size must be greater than window size")
            return -1

        kernel = np.ones(window_size, dtype=np.float32)
        rolling_sums = np.convolve(column, kernel, mode='valid')
        candidates = np.where(rolling_sums <= threshold)[0]

        if candidates.size == 0:
            return -1

        return int(candidates[0])

    def _smoothen(self, series: pd.Series, window_size: int = 100, percentile: float = 95.0) -> pd.Series:
        smoothed = series.copy()
        n = len(series) # TODO: Chunk by chunk is better?

        lower_quantile = (100 - percentile) / 200
        upper_quantile = 1 - lower_quantile

        for i in range(0, n, window_size):
            temp_min = min(i + window_size, n)
            subset_slice = slice(i, temp_min) # e.g. [i:temp_min]
            subset = series.iloc[subset_slice]

            if subset.empty:
                continue

            median = subset.median()
            lower_bound = subset.quantile(lower_quantile)
            upper_bound = subset.quantile(upper_quantile)
            outlier_mask = (subset < lower_bound) | (subset > upper_bound)
            smoothed.loc[subset[outlier_mask].index] = median

        return smoothed

    def _perform_pos_analysis(self, iteration_time_series: pd.Series, start_index: int) -> int:
        """
        Analyzes Performance Outcome Signals (POS) using a variant of the KB-KSSD algorithm.
        Phase 2 & 3 of the Hybrid Model:
        1. Defines ROI (Region of Interest) after the CAS start_index
        2. Smoothens the POS signal (Stage 1).
        3. Applies Kernel Convolution to find the 'step down' to steady state (Stage 2).
        """

        KERNEL_SIZE = DUMMY          # Size of the step-detector kernel
        SMOOTH_WINDOW = DUMMY       # Window for outlier removal

        roi_start_index = start_index # we should be guaranteed to start from at least >= 0
        LAST = None
        pos_roi = iteration_time_series.iloc[slice(roi_start_index, LAST)]


        # Validation: Ensure ROI is long enough for the kernel
        if len(pos_roi) < KERNEL_SIZE:
            return -1


        smoothed_roi = self._smoothen(pos_roi, window_size=SMOOTH_WINDOW)
        half_k = DUMMY
        kernel = np.concatenate([np.ones(half_k), -1 * np.ones(half_k)])
        convolved = np.convolve(smoothed_roi, kernel, mode='valid')
        absolute_index = roi_start_index + np.argmax(convolved) + half_k
        return int(absolute_index)

    def detect_cutoff_index(self, df: pd.DataFrame) -> int:

        if df.empty:
            logger.info("detect_cutoff_index: DataFrame is empty, returning cutoff 0.")
            return 0

        cas_series = pd.Series(df['compilation_time_ms'])
        cas_signal = self._perform_cas_analysis(
            cas_series, self.window_size, self.threshold
        )

        logger.debug(f"cas_signal: {cas_signal}")

        start_index_for_pos: int = 0
        cas_offset: int = 0

        if cas_signal >= 0: # CAS SUCCEEDED
            start_index_for_pos = cas_signal
            cas_offset = cas_signal
        else: # CAS FAILED
            logger.warning("detect_cutoff_index: CAS analysis failed. "
                            "Relying fully on POS analysis from index 0.")

        pos_series = pd.Series(df['iteration_time_ms'])
        pos_offset = self._perform_pos_analysis(
            pos_series, start_index=start_index_for_pos # Either 0 or CAS signal
        )

        logger.debug(f"pos_offset: {pos_offset}")

        if pos_offset >= 0: # POS SUCCEEDED

            if (cas_signal <= 0): # 0 or -1 (useless or failed)
                return pos_offset

            verified_index = self._verify_index_by_median_diff(
                series=pos_series,
                new_index=pos_offset,
                original_index=cas_offset,
                window_size=100
            )

            final_signal = verified_index
            logger.debug(f"final_signal: {final_signal}")
            return final_signal

        # POS signal failed lets try to fall back to CAS signal
        if cas_offset >= 0:
            logger.warning(f"detect_cutoff_index: POS analysis failed. "
                            f"Falling back to original CAS signal: {cas_offset}.")
            return cas_offset

        # Both signals failed
        logger.error("detect_cutoff_index: Both CAS and POS methods failed. "
                        "Skipping dataset (returning 0).")
        return 0

    def _get_median_diff(self, series: pd.Series, index: int, window_size: int) -> float:

            left_window = series.iloc[slice(index - window_size, index)]
            right_window = series.iloc[slice(index, index + window_size)]

            if left_window.empty or right_window.empty:
                return 0.0

            left_median = left_window.median()
            right_median = right_window.median()

            return abs(right_median - left_median)

    def _verify_index_by_median_diff(self, series: pd.Series, new_index: int, original_index: int, window_size: int) -> int:

        new_index_diff = self._get_median_diff(series, new_index, window_size)
        original_index_diff = self._get_median_diff(series, original_index, window_size)

        if new_index_diff > original_index_diff:
            return new_index

        return original_index


if __name__ == "__main__":
    pass
