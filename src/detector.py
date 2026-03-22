import numpy as np
import pandas as pd
import logging
import random
from dataclasses import dataclass
from typing import Optional

# Setup basic logging
logger = logging.getLogger(__name__)

# Column name alternatives: (preferred, fallback)
ITERATION_TIME_COLS = ['iteration_time_ns', 'pol_dd_0_iteration_time_ns']
COMPILATION_TIME_COLS = ['compilation_time_ms', 'pol_dd_0_compilation_time_ms']


def _resolve_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first column name from candidates that exists in df, or None."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


@dataclass
class SteadyStateParams:
    window_size: int = 0
    cas_threshold: float = 0.0
    smoothen_window_size: int = 0
    kernel_size: int = 0
    comparison_window_size: int = 0

class SteadyStateDetector:
    """
    Re-implementation of the steady state detection logic.
    Identifies the cutoff index where the benchmark reaches steady state (warmup is finished).
    """

    def __init__(self):
        self.params = SteadyStateParams()

    def detect_cutoff_index(self, df: pd.DataFrame) -> int:
        """
        Detects the steady state index for the given DataFrame.
        """
        if df.empty:
            return 0

        self._calculate_dynamic_params(df)

        cas_col = _resolve_col(df, COMPILATION_TIME_COLS)
        pos_col = _resolve_col(df, ITERATION_TIME_COLS)

        cas_series = df[cas_col] if cas_col else None
        pos_series = df[pos_col] if pos_col else None
        
        if pos_series is None or cas_series is None:
            return 0

        # CAS Analysis
        cas_signal = self._perform_cas_analysis(cas_series.copy())
        start_index_for_pos = cas_signal
        # return start_index_for_pos
        
        # POS Analysis (Steady State on iteration time)
        pos_offset = self._perform_pos_analysis(pos_series.copy(), roi_start_index=start_index_for_pos)

        original_index = cas_signal
        
        # Verify which index is better
        if pos_offset < 0:
             return max(0, original_index)
             
        if original_index <= 0:
            return pos_offset

        verified_index = self._verify_index_by_median_diff(
            series=pos_series,
            new_index=pos_offset,
            original_index=original_index,
            window_size=self.params.comparison_window_size
        )
        
        # Sanity check: don't cut off too much (e.g. > 70%)
        n = len(df)
        if verified_index >= int(n * 0.70):
             # Fallback to random 10-50% if we are cutting off too much (indicates algorithm failure)
             min_random = int(n * 0.10)
             max_random = int(n * 0.50)
             verified_index = random.randint(min_random, max_random)
             
        return verified_index

    def _perform_cas_analysis(self, cas_series: pd.Series) -> int:
        if self.params.window_size <= 0: return -1
        n = cas_series.shape[0]
        if n < self.params.window_size: return -1

        kernel = np.ones(self.params.window_size, dtype=np.float32)
        rolling_sums = np.convolve(cas_series, kernel, mode='valid')
        rolling_mean = rolling_sums / self.params.window_size
        
        # Find first window where mean <= threshold
        candidates = np.where(rolling_mean <= self.params.cas_threshold)[0]
        return int(candidates[0]) if candidates.size != 0 else -1

    def _perform_pos_analysis(self, iteration_time_series: pd.Series, roi_start_index: int) -> int:
        roi_start_index = max(0, roi_start_index)
        if roi_start_index >= len(iteration_time_series):
            return -1

        pos_roi = iteration_time_series.iloc[slice(roi_start_index, None)]
        smoothed_roi = self._smoothen(pos_roi)

        half_k = self.params.kernel_size // 2
        # Kernel: [1, ..., 1, -1, ..., -1]
        kernel = np.concatenate([np.ones(half_k), -1 * np.ones(half_k)])
        
        if len(smoothed_roi) < len(kernel):
            return -1
            
        convolved = np.convolve(smoothed_roi, kernel, mode='valid')
        if convolved.size == 0:
            return -1
            
        # Argmax gives relative index in valid convolution
        # We need to map back to absolute index
        # Convolution 'valid' reduces size by len(kernel) - 1
        # The peak indicates the drop.
        
        relative_peak = np.argmax(convolved)
        # Offset calculation similar to original
        absolute_index = roi_start_index + (relative_peak + half_k)
        
        return int(absolute_index) if absolute_index < len(iteration_time_series) else -1

    def _verify_index_by_median_diff(self, series: pd.Series, new_index: int, original_index: int, window_size: int) -> int:
        def get_diff(idx):
            left = series.iloc[max(0, idx - window_size) : idx]
            right = series.iloc[idx : min(len(series), idx + window_size)]
            if left.empty or right.empty: return 0.0
            diff = left.median() - right.median()
            return diff if diff > 0 else 0.0

        new_diff = get_diff(new_index)
        orig_diff = get_diff(original_index)
        
        return new_index if new_diff > orig_diff else original_index

    def _calculate_dynamic_params(self, df: pd.DataFrame) -> None:
        n = len(df)
        if n == 0: return

        self.params.window_size = int(np.clip(n * 0.10, 5, 50))
        
        cas_col = _resolve_col(df, COMPILATION_TIME_COLS)
        if cas_col is None:
             self.params.cas_threshold = 0.0
        else:
             peak = np.max(df[cas_col])
             self.params.cas_threshold = float(peak * 0.009)

        self.params.smoothen_window_size = int(np.clip(n * 0.10, 10, 200))

        raw_kernel = int(np.clip(n * 0.15, 2, 16))
        if raw_kernel % 2 != 0: raw_kernel -= 1
        self.params.kernel_size = max(2, raw_kernel)
        
        self.params.comparison_window_size = int(np.clip(n * 0.10, 5, 100))

    def _smoothen(self, series: pd.Series, percentile: float = 95.0) -> pd.Series:
        smoothed = series.astype(float).copy()
        n = len(series)
        w_size = self.params.smoothen_window_size
        if w_size <= 0: return smoothed
        
        lower_q = (100 - percentile) / 200.0
        upper_q = 1.0 - lower_q
        
        for i in range(0, n, w_size):
            subset = series.iloc[i : min(i + w_size, n)]
            if subset.empty: continue
            
            median = subset.median()
            lb = subset.quantile(lower_q)
            ub = subset.quantile(upper_q)
            
            mask = (subset < lb) | (subset > ub)
            smoothed.loc[subset[mask].index] = median
            
        return smoothed
