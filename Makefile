PYTHON = python3
PLOTS_DIR = plots

.PHONY: run clean help

run:
	$(PYTHON) src/main.py

clean:
	@echo "Cleaning plots directory..."
	rm -f $(PLOTS_DIR)/*.png
	@echo "Re-setting logger"
	rm app.log && touch app.log
	rm -rf src/__pycache__
	@echo "Clean complete."

help:
	@echo "Available commands:"
	@echo "  make run       - Run src/main.py"
	@echo "  make clean     - Remove image files from $(PLOTS_DIR)"
