#!/usr/bin/env python3

from pathlib import Path
from typing import Optional

import pandas as pd

CSV_FILENAMES = ['default.csv.one-per-rep.csv', 'default.csv']
ITERATION_TIME_COLS = ['pol_dd_0_iteration_time_ns', 'iteration_time_ns']
COMPILATION_TIME_COLS = ['compilation_time_ms', 'pol_dd_0_compilation_time_ms']

PAD_VALUE = -999.0
ROOT = Path(__file__).resolve().parent.parent


ALLOWED_PLATFORM_TYPES = {
    12: "graal-ee-master-jdk-11",
    14: "graal-ee-release-jdk-11",
    15: "graal-ee-release-jdk-8",
    16: "graal-ce-master-jdk-11",
    26: "graal-ce-master-jdk-17",
    27: "graal-ee-master-jdk-17",
    28: "graal-ee-release-jdk-17",
}

ALLOWED_GC_CONFIGS = [34, 35, 43]

ALLOWED_MACHINE_HOSTS = [
    8, 9, 10, 11, 12, 13, 14, 15,
    24, 25, 26, 27, 28, 29, 30, 31,
    43, 63,
]

METADATA_SUBDIRS = {
    "benchmark_workload": "metadata/benchmark_workload/metadata",
    "platform_installation": "metadata/platform_installation/metadata",
}

def find_csv(run_dir: Path) -> Optional[Path]:
    for name in CSV_FILENAMES:
        p = run_dir / name
        if p.exists():
            return p
    return None


def resolve_iter_col(df: pd.DataFrame) -> Optional[str]:
    for col in ITERATION_TIME_COLS:
        if col in df.columns:
            return col
    return None
