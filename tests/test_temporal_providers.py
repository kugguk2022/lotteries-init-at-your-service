from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lotteries_core.likely_set_generator import (
    GameConfig,
    build_comatrices,
    detect_columns,
    observed_poi_series,
    score_ticket,
)
from lotteries_core.protocol import GameSpec
from lotteries_core.providers.temporal import (
    SCORE_SERIES_SEMANTICS,
    TEMPORAL_METHOD_VERSION,
    GarchMarkovBranchProvider,
    _build_sequence_model,
    _score_series,
)


def _history(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        [
            {
                **{f"ball_{i + 1}": value for i, value in enumerate(sorted(rng.choice(12, 4, replace=False) + 1))},
                "star_1": int(rng.integers(1, 5)),
            }
            for _ in range(rows)
        ]
    )


def test_garch_markov_branch_returns_a_reproducible_legal_budget():
    spec = GameSpec("small", 12, 4, 4, 1)
    history = _history()
    first = GarchMarkovBranchProvider().fit(history, spec).propose(
        spec, 7, np.random.default_rng(9)
    )
    second = GarchMarkovBranchProvider().fit(history, spec).propose(
        spec, 7, np.random.default_rng(9)
    )
    assert first.tickets == second.tickets
    assert len(first.tickets) == len(set(first.tickets)) == 7
    assert first.diagnostics["variance_next"] > 0
    for ticket in first.tickets:
        spec.validate_ticket(ticket)


def test_sequence_transformer_is_a_real_optional_provider():
    torch = pytest.importorskip("torch")
    del torch
    from lotteries_core.providers.temporal import SequenceTransformerProvider

    spec = GameSpec("small", 12, 4, 4, 1)
    result = SequenceTransformerProvider(epochs=1).fit(_history(), spec).propose(
        spec, 3, np.random.default_rng(7)
    )
    assert len(result.tickets) == 3
    assert result.diagnostics["epochs"] == 1
    assert result.diagnostics["method_version"] == TEMPORAL_METHOD_VERSION
    assert result.diagnostics["score_series_semantics"] == SCORE_SERIES_SEMANTICS
    assert result.diagnostics["positional_encoding"] == "sinusoidal_v1"
    for ticket in result.tickets:
        spec.validate_ticket(ticket)


@pytest.mark.parametrize(
    ("spec", "self_pairs"),
    [
        (GameSpec("main-only", 50, 5), 10),
        (GameSpec.euromillions(), 21),
        (GameSpec("nl-lotto", 45, 6), 15),
        (GameSpec.eurodreams(), 21),
    ],
)
def test_training_labels_exclude_exact_self_pairs_without_changing_candidate_matrices(
    spec, self_pairs
):
    row = {f"ball_{i}": i for i in range(1, spec.main_k + 1)}
    row.update({f"star_{i}": i for i in range(1, spec.star_k + 1)})
    history = pd.DataFrame([row, row, row])
    matrices, labels = _score_series(history, spec)
    # A row sees the other two identical draws, while candidate matrices still see all three.
    assert labels.tolist() == [2 * self_pairs] * 3
    ticket_score = score_ticket(
        tuple(range(spec.main_k)), tuple(range(spec.star_k)), matrices, "cross"
    )
    assert ticket_score == 3 * self_pairs


def test_training_label_matches_independent_leave_one_out_matrix_score():
    history = _history(7)
    spec = GameSpec("small", 12, 4, 4, 1)
    cfg = GameConfig(spec.name, spec.main_n, spec.main_k, spec.star_n, spec.star_k)
    main_cols, star_cols = detect_columns(history, cfg)
    matrices, labels = _score_series(history, spec)
    for index, row in history.iterrows():
        without_row = build_comatrices(history.drop(index=index), cfg, main_cols, star_cols)
        expected = score_ticket(
            tuple(int(v) - 1 for v in row[main_cols]),
            tuple(int(v) - 1 for v in row[star_cols]),
            without_row,
            "cross",
        )
        assert labels[index] == expected
    # Existing observed/descriptive scores retain their original self-inclusive definition.
    descriptive = observed_poi_series(history, cfg, matrices, main_cols, star_cols, "cross")
    assert np.array_equal(descriptive, labels + 10)


def test_training_label_uses_separate_number_pools_for_cross_pairs():
    history = pd.DataFrame(
        [
            {"ball_1": 1, "ball_2": 2, "star_1": 1, "star_2": 2},
            {"ball_1": 1, "ball_2": 3, "star_1": 1, "star_2": 3},
        ]
    )
    _, labels = _score_series(history, GameSpec("tiny", 3, 2, 3, 2))
    # Only the main-1 / star-1 cross pair overlaps; no main or star pair overlaps.
    assert labels.tolist() == [1, 1]


def test_invalid_duplicate_draw_cannot_silently_break_self_pair_subtraction():
    history = _history(2)
    history.loc[0, "ball_2"] = history.loc[0, "ball_1"]
    with pytest.raises(ValueError, match="distinct"):
        _score_series(history, GameSpec("small", 12, 4, 4, 1))


def test_provider_only_uses_explicit_training_snapshot_not_withheld_or_mutated_rows():
    history = _history(20)
    spec = GameSpec("small", 12, 4, 4, 1)
    provider = GarchMarkovBranchProvider().fit(history.iloc[:-1], spec)
    first = provider.propose(spec, 3, np.random.default_rng(9))
    # This would invalidate every draw if the provider re-read the caller's full data frame.
    history.loc[:, "ball_1"] = 999
    second = provider.propose(spec, 3, np.random.default_rng(9))
    assert first.tickets == second.tickets
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics["history_points"] == 19


def test_transformer_distinguishes_prior_lag_permutations_with_final_token_fixed():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = _build_sequence_model(10).eval()
    chronological = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
    permuted = chronological[:, [8, 7, 6, 5, 4, 3, 2, 1, 0, 9], :]
    with torch.no_grad():
        first = model(chronological)
        repeated = model(chronological)
        changed = model(permuted)
    assert torch.equal(first, repeated)
    assert abs(float(first.item()) - float(changed.item())) > 1e-6
    torch.manual_seed(0)
    rebuilt = _build_sequence_model(10).eval()
    with torch.no_grad():
        assert torch.equal(first, rebuilt(chronological))
