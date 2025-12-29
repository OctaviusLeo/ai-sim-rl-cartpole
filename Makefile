# Automation for common development tasks
.PHONY: help install install-dev train train-quick evaluate compare tensorboard clean

help:
	@echo "Available commands:"
	@echo "  make install          - Install package in development mode"
	@echo "  make install-dev      - Install package with development dependencies"
	@echo "  make train            - Run training with default config"
	@echo "  make train-quick      - Run quick test training (10k timesteps)"
	@echo "  make evaluate         - Evaluate the most recent model"
	@echo "  make compare          - Compare most recent model against baselines"
	@echo "  make tensorboard      - Launch TensorBoard for most recent run"
	@echo "  make clean            - Remove temporary files and caches"

install:
	pip install -e .

install-dev:
	pip install -e .
	pip install black flake8 pytest

train:
	python src/train.py --config configs/default.yaml

train-quick:
	python src/train.py --config configs/quick_test.yaml

evaluate:
	@LATEST_RUN=$$(ls -t outputs/runs | grep "^20" | head -n1); \
	if [ -z "$$LATEST_RUN" ]; then \
		echo "No runs found in outputs/runs"; \
		exit 1; \
	fi; \
	echo "Evaluating model from $$LATEST_RUN..."; \
	python src/evaluate.py --model-path outputs/runs/$$LATEST_RUN/model.zip --episodes 20 --num-seeds 3

compare:
	@LATEST_RUN=$$(ls -t outputs/runs | grep "^20" | head -n1); \
	if [ -z "$$LATEST_RUN" ]; then \
		echo "No runs found in outputs/runs"; \
		exit 1; \
	fi; \
	echo "Comparing model from $$LATEST_RUN..."; \
	python src/compare.py --model-path outputs/runs/$$LATEST_RUN/model.zip --episodes 20 --num-seeds 3

tensorboard:
	@LATEST_RUN=$$(ls -t outputs/runs | grep "^20" | head -n1); \
	if [ -z "$$LATEST_RUN" ]; then \
		echo "No runs found in outputs/runs"; \
		exit 1; \
	fi; \
	echo "Launching TensorBoard for $$LATEST_RUN..."; \
	tensorboard --logdir outputs/runs/$$LATEST_RUN/tensorboard

clean:
	rm -rf __pycache__ src/__pycache__ .pytest_cache
	rm -rf *.egg-info build dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Clean complete."
