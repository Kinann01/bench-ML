#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from logger import logger


def _save_timeseries_plot(
    series: pd.Series,
    title: str,
    save_path: Path,
    cas_cutoff_index: Optional[int] = None,
    pos_cutoff_index: Optional[int] = None,
):

    if cas_cutoff_index is None and pos_cutoff_index is None: # TODO: Check this before calling _save_timeseries_plot()
        logger.warning(
            "Neither cas_cutoff_index nor pos_cutoff_index was provided. No cutoff lines will be drawn."
        )

    fig, ax = plt.subplots(figsize=(12, 6))
    series.plot(ax=ax, label="Series Data")

    def _draw_cutoff_line(
            cutoff_index: Optional[int], color: str, label_prefix: str
        ):
            if cutoff_index is not None: # TODO: out of range indices and negative indices should be maybe covered?
                n = len(series) # TODO: chunk by chunk is better?
                pos = cutoff_index

                if 0 <= pos < n:
                    x_value = series.index[pos]
                    ax.axvline(
                        x=x_value,
                        color=color,
                        linestyle="--",
                        linewidth=1.5,
                        label=f"{label_prefix} Cutoff (Index {cutoff_index})",
                        zorder=5
                    )
                else:
                    logger.warning(
                        "%s cutoff_index %d is out of range (must be non-negative and less than %d). No cutoff line will be drawn.",
                        label_prefix, cutoff_index, n
                    )

    _draw_cutoff_line(cas_cutoff_index, "red", "Causal")
    _draw_cutoff_line(pos_cutoff_index, "green", "Post-Causal")

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    try:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Logged plot: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {save_path}: {e}")
    finally:
        plt.close(fig)


class Plotter:
    def __init__(self, save_dir: Path):
        self.save_dir : Path = save_dir # The directory where plots will be saved
        self.context_name : Path = Path() # The default.csv we are currently processing

        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DebugPlotter initialized. Plots will save to {self.save_dir}")

    def set_context(self, name: Path):
        self.context_name = name / "default.csv"

    def log_step(
        self,
        series: pd.Series,
        step_name: str,
        cas_cutoff_index: Optional[int] = None,
        pos_cutoff_index: Optional[int] = None,
    ):

        if series.empty:
            logger.warning(f"Skipping plot for {step_name}, data is empty.")
            return

        # Check if at least one index is set
        if cas_cutoff_index is None and pos_cutoff_index is None:
            logger.warning(
                "In Plotter.log_step for '%s': Neither cas_cutoff_index nor pos_cutoff_index was provided.",
                step_name
            )

        filename = f"{self.context_name}_{step_name}.png"
        save_path = self.save_dir / filename
        title = f"{self.context_name}: {step_name}"

        _save_timeseries_plot(
            series,
            title,
            save_path,
            cas_cutoff_index=cas_cutoff_index,
            pos_cutoff_index=pos_cutoff_index
        )

if __name__ == "__main__":
    pass
