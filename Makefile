download-data:
	uv run python -m src.data.download

download-instreg:
	uv run python -m src.data.download --instreg

train:
	uv run python -m src.model.training_single

train-optuna:
	uv run python -m src.model.training_optuna

train-tabpfn:
	uv run python -m src.model.training_tabpfn

train-tabpfn-hpo:
	uv run python -m src.model.training_tabpfn_hpo

train-tabpfn-rf:
	uv run python -m src.model.training_tabpfn_rf

# Get the most recent run_id from models directory
LATEST_RUN_ID := $(shell ls -td models/*/ 2>/dev/null | head -1 | sed 's|models/||' | sed 's|/||')

evaluate:
	@if [ -z "$(RUN_ID)" ]; then \
		if [ -z "$(LATEST_RUN_ID)" ]; then \
			echo "Error: No RUN_ID provided and no models found. Usage: make evaluate RUN_ID=<run_id> [SPLIT=<split>] [LIMIT=<limit>]"; \
			exit 1; \
		else \
			echo "Using most recent run_id: $(LATEST_RUN_ID)"; \
			SPLIT=$${SPLIT:-val}; \
			if [ -n "$(LIMIT)" ]; then \
				uv run python -m src.model.evaluate --run_id $(LATEST_RUN_ID) --split $$SPLIT --limit $(LIMIT); \
			else \
				uv run python -m src.model.evaluate --run_id $(LATEST_RUN_ID) --split $$SPLIT; \
			fi; \
		fi; \
	else \
		SPLIT=$${SPLIT:-val}; \
		if [ -n "$(LIMIT)" ]; then \
			uv run python -m src.model.evaluate --run_id $(RUN_ID) --split $$SPLIT --limit $(LIMIT); \
		else \
			uv run python -m src.model.evaluate --run_id $(RUN_ID) --split $$SPLIT; \
		fi; \
	fi



evaluate-test:
	@if [ -z "$(RUN_ID)" ]; then \
		if [ -z "$(LATEST_RUN_ID)" ]; then \
			echo "Error: No RUN_ID provided and no models found. Usage: make evaluate-test RUN_ID=<run_id> [LIMIT=<limit>]"; \
			exit 1; \
		else \
			echo "Using most recent run_id: $(LATEST_RUN_ID)"; \
			if [ -n "$(LIMIT)" ]; then \
				uv run python -m src.model.evaluate --run_id $(LATEST_RUN_ID) --split test --limit $(LIMIT); \
			else \
				uv run python -m src.model.evaluate --run_id $(LATEST_RUN_ID) --split test; \
			fi; \
		fi; \
	else \
		if [ -n "$(LIMIT)" ]; then \
			uv run python -m src.model.evaluate --run_id $(RUN_ID) --split test --limit $(LIMIT); \
		else \
			uv run python -m src.model.evaluate --run_id $(RUN_ID) --split test; \
		fi; \
	fi

evaluate-val:
	@if [ -z "$(RUN_ID)" ]; then \
		if [ -z "$(LATEST_RUN_ID)" ]; then \
			echo "Error: No RUN_ID provided and no models found. Usage: make evaluate-val RUN_ID=<run_id> [LIMIT=<limit>]"; \
			exit 1; \
		else \
			echo "Using most recent run_id: $(LATEST_RUN_ID)"; \
			if [ -n "$(LIMIT)" ]; then \
				uv run python -m src.model.evaluate --run_id $(LATEST_RUN_ID) --split val --limit $(LIMIT); \
			else \
				uv run python -m src.model.evaluate --run_id $(LATEST_RUN_ID) --split val; \
			fi; \
		fi; \
	else \
		if [ -n "$(LIMIT)" ]; then \
			uv run python -m src.model.evaluate --run_id $(RUN_ID) --split val --limit $(LIMIT); \
		else \
			uv run python -m src.model.evaluate --run_id $(RUN_ID) --split val; \
		fi; \
	fi

evaluate-val-1000:
	@if [ -z "$(LATEST_RUN_ID)" ]; then \
		echo "Error: No models found. Please train a model first."; \
		exit 1; \
	else \
		echo "Using most recent run_id: $(LATEST_RUN_ID)"; \
		uv run python -m src.model.evaluate --run_id $(LATEST_RUN_ID) --split val --limit 1000; \
	fi

evaluate-train:
	@if [ -z "$(RUN_ID)" ]; then \
		if [ -z "$(LATEST_RUN_ID)" ]; then \
			echo "Error: No RUN_ID provided and no models found. Usage: make evaluate-train RUN_ID=<run_id> [LIMIT=<limit>]"; \
			exit 1; \
		else \
			echo "Using most recent run_id: $(LATEST_RUN_ID)"; \
			if [ -n "$(LIMIT)" ]; then \
				uv run python -m src.model.evaluate --run_id $(LATEST_RUN_ID) --split train --limit $(LIMIT); \
			else \
				uv run python -m src.model.evaluate --run_id $(LATEST_RUN_ID) --split train; \
			fi; \
		fi; \
	else \
		if [ -n "$(LIMIT)" ]; then \
			uv run python -m src.model.evaluate --run_id $(RUN_ID) --split train --limit $(LIMIT); \
		else \
			uv run python -m src.model.evaluate --run_id $(RUN_ID) --split train; \
		fi; \
	fi

submit:
	@if [ -z "$(RUN_ID)" ]; then \
		if [ -z "$(LATEST_RUN_ID)" ]; then \
			echo "Error: No RUN_ID provided and no models found. Usage: make submit RUN_ID=<run_id>"; \
			exit 1; \
		else \
			echo "Using most recent run_id: $(LATEST_RUN_ID)"; \
			uv run python -m src.model.evaluate --run_id $(LATEST_RUN_ID) --submit; \
		fi; \
	else \
		uv run python -m src.model.evaluate --run_id $(RUN_ID) --submit; \
	fi

viz:
	uv run python -m src.viz.run_viz

format:
	uvx ruff format

calculate-distances:
	uv run python -m src.data.external_data

calculate-distances-geodesic:
	uv run python -m src.data.external_data --geodesic

install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	source ~/.bashrc
	uv --version