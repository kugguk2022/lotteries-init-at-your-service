.DEFAULT_GOAL := help

.PHONY: help venv install install-dev lint test providers benchmark roi-report e2e package check db-refresh db-check serve

PYTHON ?= python3
DB ?= data/lotteries.db
GAME ?= euromillions
HISTORY ?= data/euromillions.csv
LEDGER ?= ledger/euromillions
OUT ?= outputs/euromillions/competition_benchmark.json
BUDGET ?= 25
HOLDOUT ?= 20

help: ## Show supported developer and user commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Create a local virtual environment
	$(PYTHON) -m venv .venv
	@echo "Activate .venv, then run: make install-dev"

install: ## Install the lightweight LottoBench library locally
	$(PYTHON) -m pip install -e .

install-dev: ## Install repository test, API, ML, and release tooling
	$(PYTHON) -m pip install -e ".[dev,api,ml,repo-test,release]"

lint: ## Run repository-wide static checks
	$(PYTHON) -m ruff check .

test: ## Run the repository test suite
	$(PYTHON) -m pytest -q --maxfail=1 --disable-warnings

providers: ## List the 12 registered inference providers and availability
	$(PYTHON) -c "from lotteries_core.registry import PROVIDERS, available; ready=set(available()); [print(f'{name:32} {\"available\" if name in ready else \"optional dependency missing\"}') for name in PROVIDERS]"

benchmark: ## Run all registered providers forward-only at equal budget
	$(PYTHON) -m lotteries_core.benchmark --history $(HISTORY) --game $(GAME) --budget $(BUDGET) --holdout $(HOLDOUT) --all-providers --out $(OUT)

roi-report: ## Retrieve cumulative realized user ROI from the prospective ledger
	$(PYTHON) -m lotteries_core.outcome_tracker report --ledger $(LEDGER)

e2e: ## Validate provider registry, benchmark/ROI metrics, storage, and provenance offline
	$(PYTHON) -m scripts.validate_user_journey

package: ## Build and validate wheel and source distributions
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*
	$(PYTHON) scripts/check_distribution.py dist/*

check: lint test e2e package ## Run the complete pull-request/release gate

db-refresh: ## Refresh one game in the ignored local SQLite database (network)
	$(PYTHON) scripts/refresh_history.py --out $(DB) --game $(GAME)

db-check: ## Check local database provenance and staleness
	$(PYTHON) scripts/refresh_history.py --check --out $(DB) --game $(GAME)

serve: ## Start the optional local API on 127.0.0.1:8007
	$(PYTHON) -m lotteries_core.api
