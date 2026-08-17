.PHONY: venv install lint test draws quality branch-hmm-v3 branch-hmm-v4 lawyer

PYTHON ?= python3

venv:
	$(PYTHON) -m venv .venv

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	ruff check .
	pytest -q

quality: test

draws:
	$(PYTHON) -m euromillions.get_draws --out data/euromillions.csv --append

branch-hmm-v3:
	$(PYTHON) -m euromillions.branch_hmm_v3 --source real --history data/euromillions.csv --out-dir outputs/euromillions/branch_hmm_v3

branch-hmm-v4:
	$(PYTHON) -m euromillions.branch_hmm_v4 --source real --history data/euromillions.csv --out-dir outputs/euromillions/branch_hmm_v4

lawyer:
	$(PYTHON) -c "import importlib, pathlib, sys; print('python_executable=', sys.executable); print('python_version=', sys.version.replace(chr(10), ' ')); mod = importlib.import_module('euromillions.branch_hmm_v4'); print('branch_hmm_v4_import=', getattr(mod, '__file__', '<unknown>')); history = pathlib.Path('data/euromillions.csv'); print('history_exists=', history.exists(), history.resolve() if history.exists() else history); out_dir = pathlib.Path('outputs/euromillions'); out_dir.mkdir(parents=True, exist_ok=True); print('output_dir_ready=', out_dir.resolve())"
