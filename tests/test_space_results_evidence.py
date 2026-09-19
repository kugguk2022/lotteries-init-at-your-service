import ast
import importlib.util
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "publishing/huggingface-space/results_evidence.py"
    spec = importlib.util.spec_from_file_location("results_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_containment_requires_mains_and_stars_on_same_ticket():
    frame = pd.DataFrame([
        {"agent": "a", "contest_number": 1, "main_hits": 5, "auxiliary_hits": 0},
        {"agent": "a", "contest_number": 1, "main_hits": 0, "auxiliary_hits": 2},
        {"agent": "a", "contest_number": 2, "main_hits": 5, "auxiliary_hits": 2},
        {"agent": "a", "contest_number": 2, "main_hits": 5, "auxiliary_hits": 2},
    ])
    row = _module().containment_table(frame, 5, 2).iloc[0]
    assert row["Exact full-ticket containment"] == "1/2"
    assert row["Exact main-set containment"] == "2/2"
    assert row["Settled cash ROI"].startswith("UNASSESSED")


def test_space_theme_keeps_evidence_tables_readable_in_both_host_modes():
    path = Path(__file__).resolve().parents[1] / "publishing/huggingface-space/app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    theme = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "THEME" for t in node.targets))
    colors = {kw.arg: ast.literal_eval(kw.value) for kw in theme.keywords}

    def luminance(hex_color):
        rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return sum(c * weight for c, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

    for suffix in ("", "_dark"):
        foreground = luminance(colors["table_text_color" + suffix])
        for token in ("table_even_background_fill", "table_odd_background_fill", "table_row_focus"):
            background = luminance(colors[token + suffix])
            assert (foreground + 0.05) / (background + 0.05) >= 4.5
