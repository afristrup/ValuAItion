download-data:
	uv run python -m src.data.download

train:
	uv run python -m src.model.training_single

train-optuna:
	uv run python -m src.model.training_optuna

evaluate:
	uv run python -m src.model.evaluate --run_id 321e5bf0-846a-43c9-8bdf-1b7db4b41293 --split val