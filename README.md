# bench-ML

Anomaly detection for performance analysis of benchmarks using TS2Vec contrastive time series embeddings.

## Overview

This project detects performance anomalies in JVM benchmarks by learning temporal representations of post-warmup iteration times using [TS2Vec](https://arxiv.org/abs/2106.10466) (contrastive learning). Each benchmark version is encoded into a 320-dimensional embedding, and anomalies are flagged when consecutive versions have high cosine distance in the embedding space.

## Pipeline

```
0. loader.py         →  Load + preprocess all data → training_data_all_years.npy
1. build_index.py    →  Scan measurement dirs → run_index.pkl
2. classify_configs.py →  Classify configs as long (≥100 timesteps) or short
3. train_long.py     →  Train TS2Vec encoder on long sequences
4. analyze_configs.py →  Per-config anomaly detection → reports/
```

### Quick Start

```bash
make loader     # Load + preprocess all data → .npy
make index      # Build index from measurement data
make classify   # Classify configs
make train      # Train the encoder
make analyze    # Run anomaly detection
```

See the [Makefile](Makefile) for full command arguments and usage notes.

## Source Files (`src/`)

| File | Purpose |
|------|---------|
| `detector.py` | Steady State Detector (SSD) — identifies warmup cutoff |
| `encoder.py` | TS2Vec model — dilated CNN encoder with contrastive learning |
| `loader.py` | Data loading, SSD application, log-IQR normalization, directory scanning |
| `build_index.py` | Builds persistent Config → runs index |
| `classify_configs.py` | Classifies configs as long/short by post-SSD length |
| `analyze_configs.py` | Main anomaly detection pipeline + report generation |
| `train_long.py` | TS2Vec training script for long sequences |


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
- `tsne.png` — t-SNE embedding visualization
- `distances.csv` — consecutive version distances
- `anomaly_*.png` — detailed plots for each flagged transition