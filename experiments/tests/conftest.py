import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENTS.parent
for path in (EXPERIMENTS, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
