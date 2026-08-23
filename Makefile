.DEFAULT_GOAL := help

.PHONY: help venv install setup setup-experiments doctor lint test test-experiments providers benchmark roi-report e2e package check serve

PYTHON ?= python3
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
	@echo "Activate .venv, then run: make setup"

install: ## Install the lightweight LottoBench library locally
	$(PYTHON) -m pip install -e .

setup: ## Install LottoBench core, API, and test tooling
	$(PYTHON) -m pip install -e ".[dev,api,release]"

setup-experiments: ## Install heavy dependencies for the archived research suite
	$(PYTHON) -m pip install -e ".[dev,api,ml,repo-test,release]"

doctor: ## Verify that the selected Python can run the core development suite
	$(PYTHON) scripts/doctor.py

lint: ## Run repository-wide static checks
	$(PYTHON) -m ruff check .

test: doctor ## Run tests for the packages shipped to PyPI
	$(PYTHON) -m pytest -q --maxfail=1 --disable-warnings

test-experiments: ## Run the separate legacy research suite
	$(PYTHON) -m pytest -q experiments/tests --maxfail=1 --disable-warnings

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

check: lint test e2e package ## Run the complete core pull-request/release gate

serve: ## Start the optional local API on 127.0.0.1:8007
	$(PYTHON) -m lotteries_core.api
