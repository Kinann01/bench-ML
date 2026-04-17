# bench-ML

Anomaly detection for performance analysis of JVM benchmarks using TS2Vec contrastive time series embeddings.

## Overview

This project detects performance anomalies in JVM benchmarks by learning temporal representations of post-warmup iteration times using [TS2Vec](https://arxiv.org/abs/2106.10466) (contrastive learning). Each benchmark measurement is encoded into an embedding vector, and anomalies are flagged when consecutive compiler versions produce substantially different embeddings in the learned representation space.

## Pipeline

```
build_index.py           →  index.pkl
prepare_training_data.py  →  data/training_data.npy
train_long.py             →  models/ts2vec.pt
model_diagnostics.py      →  diagnostics/
classify_configs.py       →  configs/configs.json
analyze_configs.py        →  reports/
```

## Quick Start

```bash
# 1. Build index from dataset (supports multiple base dirs)
make index BASE_DIR=/path/to/data/2020
# or: python3 src/build_index.py --base-dir /data/2020 /data/2021 /data/2022

# 2. Prepare training data
make prepare

# 3. Train encoder
make train

# 4. Validate encoder (t-SNE + nearest neighbors)
make diagnostics

# 5. Select configs with sufficient sequence length
make classify MIN_LENGTH=100

# 6. Run anomaly detection
make analyze
```

See `make help` for all available commands and configurable variables.

## Configuration

All numerical parameters (SSD thresholds, anomaly flagging criteria, training
hyperparameters) have built-in defaults and can be overridden via
[pipeline.conf](pipeline.conf).

## Project Structure

```
src/
  config.py                 Config dataclass
  constants.py              Shared constants and helpers
  pipeline_config.py        Configuration file reader
  detector.py               Steady-state detection (CAS)
  encoder.py                TS2Vec encoder + contrastive loss
  loader.py                 Dataset scanner + metadata resolver
  build_index.py            Index builder
  prepare_training_data.py  Training data preparation
  train_long.py             Encoder training
  model_diagnostics.py      Encoder validation (t-SNE, neighbors)
  classify_configs.py       Config selection by sequence length
  analyze_configs.py        Per-config anomaly detection + reporting

pipeline.conf               Adjustable pipeline parameters
Makefile                    Pipeline automation
```

## Config Format

Configs are JSON arrays specifying benchmark configurations:

```json
[
  {
    "benchmark_type": "batik-small",
    "machine_host": 10,
    "platform_type": 12,
    "gc_config": 34
  }
]
```

## Reports

Each analyzed config generates a report directory under `reports/`:
- `report.txt` — text summary with stats and flagged anomalies
- `tsne.png` — t-SNE embedding visualization colored by version order
- `distances.csv` — all consecutive version distances with flagged status
- `anomaly_*.png` — raw iteration time plots for each flagged transition

Additionally, the pipeline generates:
- `analysis_summary.json` — full analysis results for all configs, including flagged version transitions with distances and z-scores
- `cross_config_report.csv` — cross-configuration corroboration results

Version changes flagged in `distances.csv` can be cross-referenced with `analysis_summary.json` for full details.

## Dataset

Uses the [GraalVM Compiler Benchmark Results Dataset](https://zenodo.org/communities/graalvm-compiler-benchmark-results) (Bulej et al., ICPE 2023). Expects base directory with `measurement/` and `metadata/` subdirectories.

## Requirements

Python 3.9+

```bash
pip install -r requirements.txt
```
