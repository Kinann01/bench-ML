PYTHON = python3
SRC = src

.PHONY: index classify train analyze clean help

# Build index: --base-dir is the directory containing the years of measurement subdirectories along with the 
# respect metadata (i.e. base-dir/2023/, base-dir/2024/, etc.) where each year directory contains measurement/ and metadata/)
index:
	$(PYTHON) $(SRC)/build_index.py --base-dir .

# Classify configs into short / long based on threshhold given an index db and which configs we are interested in
classify:
	$(PYTHON) $(SRC)/classify_configs.py --configs configs/configs_verified.json --index index.pkl --threshold 100

# Train TS2Vec encoder on the long configs (training_data_long.npy) with 50 --epochs and default other params
train:
	$(PYTHON) $(SRC)/train_long.py --data training_data_long.npy --epochs 50

# Analyze configs using the trained model and index, outputting reports to reports/ directory
analyze:
	$(PYTHON) $(SRC)/analyze_configs.py \
		--configs configs/configs_long.json \
		--index run_index.pkl \
		--model TS2Vec/ts2vec_long.pt \
		--output-dir reports

clean:
	@echo "Cleaning generated files..."
	rm -rf reports/
	rm -rf src/__pycache__
	@echo "Clean complete."

help:
	@echo "Available commands:"
	@echo "  make index     - Build Config→runs index from measurement dirs"
	@echo "  make classify  - Classify configs as long/short"
	@echo "  make train     - Train TS2Vec encoder"
	@echo "  make analyze   - Run per-config anomaly detection"
	@echo "  make clean     - Remove generated reports and caches"
