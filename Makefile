download-data:
	uv run python -m src.data.download

train:
	uv run python -m src.model.training_single

train-optuna:
	uv run python -m src.model.training_optuna

train-tabpfn:
	uv run python -m src.model.training_tabpfn

train-tabpfn-hpo:
	uv run python -m src.model.training_tabpfn_hpo

evaluate:
	uv run python -m src.model.evaluate --run_id 321e5bf0-846a-43c9-8bdf-1b7db4b41293 --split val

viz:
	uv run python -m src.viz.run_viz

format:
	uvx ruff format

calculate-distances:
	uv run python -m src.data.external_data

calculate-distances-geodesic:
	uv run python -m src.data.external_data --geodesic