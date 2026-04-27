# bench-ML

Anomaly detection for performance analysis of JVM benchmarks using TS2Vec contrastive time series embeddings.

## Overview

This project detects performance anomalies in JVM benchmarks by learning temporal representations of post-warmup iteration times using [TS2Vec](https://arxiv.org/abs/2106.10466) (contrastive learning). Each benchmark measurement is encoded into an embedding vector, and anomalies are flagged when consecutive compiler versions produce substantially different embeddings in the learned representation space.

## Pipeline

The pipeline has two distinct workflows that use two distinct indexes so they don't collide:

**Inference** (analyse a dataset for anomalies):
```
build_index.py        →  run_index.pkl
classify_configs.py   →  configs/configs.json
analyze_configs.py    →  reports/
```

**Training** (build a new encoder from richer data — multiple years, no metadata filtering):
```
build_index.py            →  training_index.pkl   (--include-all)
prepare_training_data.py  →  data/training_data.npy
train_long.py             →  models/ts2vec.pt
model_diagnostics.py      →  diagnostics/
```

## Quick Start

### Inference (using an existing trained encoder)

```bash
# 1. Build the analysis index (strict metadata filtering)
make index BASE_DIR=/path/to/data/2020

# 2. Select configs with sufficient post-SSD sequence length
make classify MIN_LENGTH=100

# 3. Run per-config anomaly detection
make analyze
```

### Training (build a new encoder)

```bash
# 1. Build the training index from one or more datasets, without
#    filtering, so the encoder sees as much data as possible.
make index-train BASE_DIR="/data/2016 /data/2017 /data/2018 /data/2019 /data/2020"

# 2. Prepare the training data (SSD + log/IQR normalization + padding)
make prepare

# 3. Train the TS2Vec encoder
make train

# 4. Validate the encoder (t-SNE + nearest neighbors on a sample)
make diagnostics

# Then use the trained model in the inference workflow above.
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
- `report.txt` — text summary with stats, diagnostic sections, and flagged anomalies
- `distances.csv` — all consecutive version distances with flagged status
- `anomaly_*.png` — raw iteration time plots for each flagged transition

Additionally, the pipeline generates:
- `analysis_summary.json` — full analysis results for all configs, including flagged version transitions with distances and z-scores
- `cross_config_report.csv` — cross-configuration corroboration results

Corrorborated versions in `cross_config_report.csv` can be cross-referenced with `analysis_summary.json` for full details.

## Dataset

Uses the [GraalVM Compiler Benchmark Results Dataset](https://zenodo.org/communities/graalvm-compiler-benchmark-results) (Bulej et al., ICPE 2023). Expects base directory with `measurement/` and `metadata/` subdirectories.

## Requirements

Python 3.9+

```bash
pip install -r requirements.txt
```
