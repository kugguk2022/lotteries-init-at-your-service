"""Static guards for previously repaired research-script structure.

The legacy modules perform expensive work at import time, so these checks use
the AST instead of importing them.  They protect the two repairs that must not
regress while the public package API is restored.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RESEARCH_SCRIPTS = (
    Path("euromillions/roi.py"),
    Path("euromillions_agent/lotto_lab.py"),
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


@pytest.mark.parametrize("path", RESEARCH_SCRIPTS)
def test_train_grok_keeps_a_value_return_on_every_terminal_path(path: Path):
    functions = _functions(path)
    train_grok = functions.get("train_grok")

    assert train_grok is not None, f"{path} lost train_grok"
    value_returns = [
        node
        for node in ast.walk(train_grok)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    assert value_returns, f"{path}: train_grok no longer returns its signal and loss"
    assert isinstance(train_grok.body[-1], ast.Return), (
        f"{path}: train_grok must end with an explicit return, not a dangling expression"
    )


@pytest.mark.parametrize("path", RESEARCH_SCRIPTS)
def test_zscore_remains_a_real_function_with_an_explicit_return(path: Path):
    functions = _functions(path)
    zscore = functions.get("zscore")

    assert zscore is not None, f"{path} lost zscore"
    assert any(
        isinstance(node, ast.Return) and node.value is not None
        for node in ast.walk(zscore)
    ), f"{path}: zscore has no value-returning path"


@pytest.mark.parametrize("path", RESEARCH_SCRIPTS)
def test_research_scripts_have_no_dangling_top_level_expressions(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dangling = [
        node.lineno
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Expr)
        and not (
            index == 0
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]

    assert dangling == [], f"{path}: dangling top-level expressions at lines {dangling}"
