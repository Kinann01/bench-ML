PYTHON = python3
SRC = src
CONF = pipeline.conf

BASE_DIR ?= .
INDEX ?= run_index.pkl
MODEL ?= models/ts2vec.pt
MIN_LENGTH ?= 100

.PHONY: index prepare train diagnostics classify analyze clean help

# Step 1: Build index from base directory containing measurement/ and metadata/
index:
	$(PYTHON) $(SRC)/build_index.py --base-dir $(BASE_DIR) --output $(INDEX)

# Step 2: Prepare training data from index (SSD + log+IQR normalization + padding)
prepare:
	$(PYTHON) $(SRC)/prepare_training_data.py --index $(INDEX) --output data/training_data.npy

# Step 3: Train TS2Vec encoder
train:
	$(PYTHON) $(SRC)/train_long.py --data data/training_data.npy --output $(MODEL) --conf $(CONF)

# Step 4: Run encoder diagnostics (t-SNE + nearest neighbors)
diagnostics:
	$(PYTHON) $(SRC)/model_diagnostics.py --index $(INDEX) --model $(MODEL) --conf $(CONF)

# Step 5: Select configurations with sufficient sequence length
classify:
	$(PYTHON) $(SRC)/classify_configs.py --index $(INDEX) --min-length $(MIN_LENGTH)

# Step 6: Run per-config anomaly detection
analyze:
	$(PYTHON) $(SRC)/analyze_configs.py \
		--configs configs/configs.json \
		--index $(INDEX) \
		--model $(MODEL) \
		--conf $(CONF) \
		--output-dir reports

clean:
	rm -rf reports/ diagnostics/ plots_training/
	rm -rf src/__pycache__

help:
	@echo "Pipeline:"
	@echo "  make index        BASE_DIR=/data/2020      - Build measurement index"
	@echo "  make prepare                               - Prepare training data from index"
	@echo "  make train                                 - Train TS2Vec encoder"
	@echo "  make diagnostics                           - Run encoder diagnostics (t-SNE, neighbors)"
	@echo "  make classify     MIN_LENGTH=100           - Select configs by min sequence length"
	@echo "  make analyze                               - Run anomaly detection pipeline"
	@echo "  make clean                                 - Remove generated outputs"
	@echo ""
	@echo "Configurable variables:"
	@echo "  BASE_DIR    Base directory with measurement/ and metadata/ (default: .)"
	@echo "  INDEX       Index pickle file (default: run_index.pkl)"
	@echo "  MODEL       Trained model path (default: models/ts2vec.pt)"
	@echo "  MIN_LENGTH  Min post-SSD length for classify (default: 100)"
	@echo "  CONF        Pipeline config file (default: pipeline.conf)"