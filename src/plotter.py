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
    verified_index: Optional[int] = None,
):

    fig, ax = plt.subplots(figsize=(12, 6))
    series.plot(ax=ax, label="Iteration Time (ms)", alpha=0.7)

    def _draw_cutoff_line(
            cutoff_index: Optional[int], color: str, label_prefix: str
        ):
            if cutoff_index is not None:
                n = len(series)

                if 0 <= cutoff_index < n:
                    x_value = series.index[cutoff_index]

                    ax.axvline(
                        x=x_value,
                        color=color,
                        linestyle="--",
                        linewidth=2.0,
                        label=f"{label_prefix} Cutoff (Idx {cutoff_index})",
                        zorder=5
                    )
                elif cutoff_index == -1:
                     logger.debug(f"{label_prefix} cutoff not found (index -1).")
                else:
                    logger.warning(
                        "%s cutoff_index %d is out of range (0 to %d). Skipping line.",
                        label_prefix, cutoff_index, n
                    )

    _draw_cutoff_line(cas_cutoff_index, "red", "CAS results")
    _draw_cutoff_line(pos_cutoff_index, "green", "POS results)")
    _draw_cutoff_line(verified_index, "blue", "Final")

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time (ns)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    try:
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        logger.info(f"Logged plot: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {save_path}: {e}")
    finally:
        plt.close(fig)


class Plotter:
    def __init__(self, save_dir: Path):
        self.save_dir: Path = Path(save_dir)
        self.context_name: str = "default"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Plotter initialized. Root plot dir: {self.save_dir}")

    def set_context(self, path_context: Path) -> None:
        """
        Sets the context name based on the file path structure.
        from .../88/97/66/8669788/default.csv, 88_97_66_8669788 becomes the context
        """
        directory = path_context.parent if path_context.suffix else path_context
        relevant_parts = directory.parts[-4:]
        self.context_name = "_".join(relevant_parts)

    def log_step(
        self,
        series: pd.Series,
        step_name: str,
        cas_cutoff_index: Optional[int] = None,
        pos_cutoff_index: Optional[int] = None,
        verified_index: Optional[int] = None,
    ) -> None:

        if series.empty:
            logger.warning(f"Skipping plot for {step_name}, data is empty.")
            return

        filename = f"{self.context_name}_{step_name}.png"
        save_path = self.save_dir / filename
        title = f"{self.context_name}: {step_name}"

        _save_timeseries_plot(
            series,
            title,
            save_path,
            cas_cutoff_index=cas_cutoff_index,
            pos_cutoff_index=pos_cutoff_index,
            verified_index=verified_index,
        )
