"""Forward-only temporal-model providers extracted from the experimental pipelines.

Both models forecast the next pair-co-occurrence score and then rank a seeded, fixed-size pool of
legal tickets by distance to that forecast.  They are benchmark entrants, not probability models
for a fair lottery draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..likely_set_generator import (
    GameConfig,
    build_comatrices,
    detect_columns,
    observed_poi_series,
    score_ticket,
)
from ..protocol import GameSpec, InferenceProvider, ProviderResult


def _score_series(history: pd.DataFrame, spec: GameSpec):
    cfg = GameConfig(spec.name, spec.main_n, spec.main_k, spec.star_n, spec.star_k)
    main_cols, star_cols = detect_columns(history, cfg)
    if len(main_cols) < spec.main_k or len(star_cols) < spec.star_k:
        raise ValueError("history does not contain enough main/star columns")
    matrices = build_comatrices(history, cfg, main_cols, star_cols)
    poi = observed_poi_series(history, cfg, matrices, main_cols, star_cols, "cross").astype(float)
    return matrices, poi


def _rank_candidate_pool(
    provider: InferenceProvider,
    spec: GameSpec,
    matrices,
    target: float,
    budget: int,
    rng: np.random.Generator,
) -> ProviderResult:
    pool_size = min(spec.n_tickets(), max(2048, budget * 80))
    pool = provider._sample_distinct_tickets(spec, pool_size, rng)
    raw = np.asarray(
        [
            score_ticket(
                tuple(v - 1 for v in main), tuple(v - 1 for v in star), matrices, "cross"
            )
            for main, star in pool
        ],
        dtype=float,
    )
    order = np.argsort(np.abs(raw - target), kind="stable")[:budget]
    tickets = [pool[int(i)] for i in order]
    scores = -np.abs(raw[order] - target)
    return ProviderResult(tickets=tickets, scores=scores)


class GarchMarkovBranchProvider(InferenceProvider):
    """GARCH conditional variance plus a two-state empirical Markov branch model."""

    name = "garch_markov_branch"
    description = (
        "Forecasts the co-occurrence-score level with a linear mean, GARCH(1,1) variance, and "
        "two-state upper/lower branch transitions before ranking a seeded legal-ticket pool."
    )

    def __init__(self, window: int = 52) -> None:
        self.window = int(window)
        self._history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame, spec: GameSpec | None = None):
        if spec is None:
            raise ValueError("GarchMarkovBranchProvider.fit requires a GameSpec")
        self._history = history.copy()
        return self

    @staticmethod
    def _forecast(values: np.ndarray, window: int) -> tuple[float, dict]:
        y = np.asarray(values[-window:], dtype=float)
        if len(y) < 12:
            target = float(np.mean(y)) if len(y) else 0.0
            return target, {"fallback": "trailing_mean", "history_points": len(y)}

        t = np.arange(len(y), dtype=float)
        slope, intercept = np.polyfit(t, y, 1)
        fitted = intercept + slope * t
        residual = y - fitted

        # Stable GARCH(1,1) filter. Fixed coefficients avoid repeatedly optimising a fragile model
        # inside every walk-forward step; the resulting variance remains fully causal.
        alpha, beta = 0.10, 0.85
        base_var = max(float(np.var(residual)), 1e-9)
        omega = (1.0 - alpha - beta) * base_var
        variance = base_var
        for err in residual:
            variance = omega + alpha * float(err * err) + beta * variance

        states = (residual >= 0.0).astype(int)
        transitions = np.ones((2, 2), dtype=float)  # Laplace smoothing
        for previous, current in zip(states[:-1], states[1:]):
            transitions[int(previous), int(current)] += 1.0
        transitions /= transitions.sum(axis=1, keepdims=True)
        probability_upper = float(transitions[int(states[-1]), 1])

        mean_next = float(intercept + slope * len(y))
        branch_shift = (2.0 * probability_upper - 1.0) * float(np.sqrt(variance))
        return mean_next + branch_shift, {
            "mean_next": mean_next,
            "variance_next": variance,
            "probability_upper": probability_upper,
            "transition_matrix": transitions.tolist(),
            "history_points": len(y),
        }

    def propose(self, spec: GameSpec, budget: int, rng: np.random.Generator) -> ProviderResult:
        if self._history is None:
            raise RuntimeError("GarchMarkovBranchProvider.propose called before fit()")
        matrices, poi = _score_series(self._history, spec)
        target, diagnostics = self._forecast(poi, self.window)
        result = _rank_candidate_pool(self, spec, matrices, target, budget, rng)
        result.diagnostics.update({"target": target, "window": self.window, **diagnostics})
        return result


class SequenceTransformerProvider(InferenceProvider):
    """Small Transformer forecast of the co-occurrence-score sequence (PyTorch optional)."""

    name = "sequence_transformer"
    description = (
        "Tiny causal Transformer over the historical co-occurrence-score sequence; its next-score "
        "forecast ranks a seeded legal-ticket pool. Requires PyTorch."
    )

    def __init__(self, sequence_length: int = 10, epochs: int = 8) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("sequence_transformer requires the 'ml' extra (PyTorch)") from exc
        self.sequence_length = int(sequence_length)
        self.epochs = int(epochs)
        self._history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame, spec: GameSpec | None = None):
        if spec is None:
            raise ValueError("SequenceTransformerProvider.fit requires a GameSpec")
        self._history = history.copy()
        return self

    def _forecast(self, values: np.ndarray) -> tuple[float, dict]:
        import torch
        from torch import nn

        y = np.asarray(values, dtype=np.float32)
        seq_len = min(self.sequence_length, max(2, len(y) // 4))
        if len(y) <= seq_len + 2:
            target = float(np.mean(y[-seq_len:])) if len(y) else 0.0
            return target, {"fallback": "trailing_mean", "history_points": len(y)}

        mean, std = float(y.mean()), max(float(y.std()), 1e-6)
        z = (y - mean) / std
        x = np.stack([z[i : i + seq_len] for i in range(len(z) - seq_len)])
        target = z[seq_len:]

        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        xt = torch.tensor(x[:, :, None], dtype=torch.float32)
        yt = torch.tensor(target, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(1, 8)
                layer = nn.TransformerEncoderLayer(
                    d_model=8, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=1)
                self.head = nn.Linear(8, 1)

            def forward(self, batch):
                encoded = self.encoder(self.embed(batch))
                return self.head(encoded[:, -1]).squeeze(-1)

        model = Model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        final_loss = float("nan")
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            prediction = model(xt)
            loss = loss_fn(prediction, yt)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())

        model.eval()
        last = torch.tensor(z[-seq_len:][None, :, None], dtype=torch.float32)
        with torch.no_grad():
            forecast_z = float(model(last).item())
        return mean + std * forecast_z, {
            "sequence_length": seq_len,
            "epochs": self.epochs,
            "training_loss": final_loss,
            "history_points": len(y),
        }

    def propose(self, spec: GameSpec, budget: int, rng: np.random.Generator) -> ProviderResult:
        if self._history is None:
            raise RuntimeError("SequenceTransformerProvider.propose called before fit()")
        matrices, poi = _score_series(self._history, spec)
        target, diagnostics = self._forecast(poi)
        result = _rank_candidate_pool(self, spec, matrices, target, budget, rng)
        result.diagnostics.update({"target": target, **diagnostics})
        return result
