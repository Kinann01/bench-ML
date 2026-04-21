#!/usr/bin/env python3

import configparser
from pathlib import Path
from typing import Optional

DEFAULTS = {
    "ssd": {
        "window_fraction": 0.10,
        "window_min": 5,
        "window_max": 50,
        "threshold_fraction": 0.009,
    },
    "anomaly_flagging": {
        "threshold_percentile": 97.0,
        "min_z_score": 2.0,
        "distance_metric": "cosine",
    },
    "reliability": {
        "mean_dist_threshold": 0.35,
        "length_ratio_threshold": 5.0,
        "min_versions": 10,
        "min_max_z_score": 1.5,
    },
    "cross_config": {
        "strong_ratio": 0.40,
        "moderate_ratio": 0.20,
        "min_flagged_for_moderate": 2,
        "min_flagged_for_weak": 2,
    },
    "training": {
        "epochs": 40,
        "batch_size": 32,
        "learning_rate": 0.001,
        "hidden_dim": 128,
        "repr_dim": 320,
        "depth": 10,
        "mask_ratio": 0.5,
        "weight_decay": 0.0001,
        "eta_min": 0.000001,
    },
}

_INT_KEYS = {"window_min", "window_max", "min_versions",
             "min_flagged_for_moderate", "min_flagged_for_weak",
             "epochs", "batch_size", "hidden_dim", "repr_dim", "depth"}

_STR_KEYS = {"distance_metric"}


class PipelineConfig:

    def __init__(self, conf_path: Optional[str] = None):

        self._data = {}
        for section, params in DEFAULTS.items():
            self._data[section] = dict(params)

        if conf_path:
            self._load(conf_path)

    def _load(self, path: str):

        p = Path(path)
        if not p.exists():
            print(f"Config file not found: {p}, using defaults")
            return

        parser = configparser.ConfigParser()
        parser.read(p)

        for section in parser.sections():
            if section not in self._data:
                continue
            for key, val in parser.items(section):
                if key in self._data[section]:
                    if key in _STR_KEYS:
                        self._data[section][key] = val.strip()
                    elif key in _INT_KEYS:
                        self._data[section][key] = int(val)
                    else:
                        self._data[section][key] = float(val)

        print(f"Loaded config from {p}")

    def get(self, section: str, key: str):
        return self._data[section][key]

    @property
    def ssd(self) -> dict:
        return self._data["ssd"]

    @property
    def anomaly_flagging(self) -> dict:
        return self._data["anomaly_flagging"]

    @property
    def reliability(self) -> dict:
        return self._data["reliability"]

    @property
    def cross_config(self) -> dict:
        return self._data["cross_config"]

    @property
    def training(self) -> dict:
        return self._data["training"]
