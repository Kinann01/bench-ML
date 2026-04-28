PYTHON = python3
SRC = src
CONF = pipeline.conf

BASE_DIR ?= .
ANALYSIS_INDEX ?= run_index.pkl
TRAINING_INDEX ?= training_index.pkl
MODEL ?= models/ts2vec.pt
ANALYSIS_MIN_LENGTH ?= 100
TRAIN_MIN_LENGTH ?= 150

.PHONY: index index-train prepare train diagnostics classify analyze clean help

# ---------------------------------------------------------------------
# Inference workflow:
#   make index BASE_DIR=2020/   ->   make diagnostics  ->
#   make classify   ->   make analyze
# ---------------------------------------------------------------------

# Step 1: Build the analysis index (strict metadata filtering).
index:
	$(PYTHON) $(SRC)/build_index.py --base-dir $(BASE_DIR) --output $(ANALYSIS_INDEX)

# Step 2: Sanity-check the trained model on the data the
# inference pipeline will actually consume (t-SNE + nearest neighbours).
diagnostics:
	$(PYTHON) $(SRC)/model_diagnostics.py --index $(ANALYSIS_INDEX) --model $(MODEL) --conf $(CONF)

# Step 3: Select configurations with sufficient sequence length.
classify:
	$(PYTHON) $(SRC)/classify_configs.py --index $(ANALYSIS_INDEX) --min-length $(ANALYSIS_MIN_LENGTH)

# Step 4: Run per-config anomaly detection.
analyze:
	$(PYTHON) $(SRC)/analyze_configs.py \
		--configs configs/configs.json \
		--index $(ANALYSIS_INDEX) \
		--model $(MODEL) \
		--conf $(CONF) \
		--output-dir reports

# ---------------------------------------------------------------------
# Training workflow:
#   make index-train BASE_DIR="2016/ ... 2020/"   ->   make prepare   ->
#   make train
# ---------------------------------------------------------------------

# Step 1 (training): Build the training index without metadata filtering.
index-train:
	$(PYTHON) $(SRC)/build_index.py --base-dir $(BASE_DIR) --output $(TRAINING_INDEX) --include-all

# Step 2 (training): Prepare training data from the training index.
prepare:
	$(PYTHON) $(SRC)/prepare_training_data.py --index $(TRAINING_INDEX) --output data/training_data.npy --min-length $(TRAIN_MIN_LENGTH)

# Step 3 (training): Train the TS2Vec encoder on the prepared data.
train:
	$(PYTHON) $(SRC)/train_long.py --data data/training_data.npy --output $(MODEL) --conf $(CONF)

clean:
	rm -rf reports/ diagnostics/ plots_training/
	rm -rf src/__pycache__

help:
	@echo "Two workflows:"
	@echo ""
	@echo "Inference (analyse a dataset for anomalies):"
	@echo "  make index        BASE_DIR=2020/                       Build analysis index"
	@echo "  make diagnostics                                       Run diagnostics on the model"
	@echo "  make classify     ANALYSIS_MIN_LENGTH=100              Select configs by length"
	@echo "  make analyze                                           Run anomaly detection"
	@echo ""
	@echo "Training (build a new encoder from richer data):"
	@echo "  make index-train  BASE_DIR=\"2016/ ... 2020/\"           Build training index"
	@echo "  make prepare      TRAIN_MIN_LENGTH=150                 Prepare training data"
	@echo "  make train                                             Train TS2Vec encoder"
	@echo ""
	@echo "  make clean                                             Remove generated outputs"
	@echo ""
	@echo "Variables:"
	@echo "  BASE_DIR              Base directory(ies) with measurement/ and metadata/ (default: .)"
	@echo "  ANALYSIS_INDEX        Analysis index pickle (default: run_index.pkl)"
	@echo "  TRAINING_INDEX        Training index pickle (default: training_index.pkl)"
	@echo "  MODEL                 Trained model path (default: models/ts2vec.pt)"
	@echo "  ANALYSIS_MIN_LENGTH   Min post-SSD length for classify (default: 100)"
	@echo "  TRAIN_MIN_LENGTH      Min post-SSD length for prepare  (default: 150)"
	@echo "  CONF                  Pipeline config file (default: pipeline.conf)"