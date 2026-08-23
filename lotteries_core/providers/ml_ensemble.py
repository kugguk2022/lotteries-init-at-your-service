"""GLM + gradient-boosting + (optional) deep-learning ensemble.

The user asked for a "GLM + XGBoost + deep learning" stack. Here it is -- but pointed at a target
where machine learning is *legitimately useful* rather than at the draw itself (which is random and
unlearnable). This ensemble learns to predict a number's **crowd-popularity / coverage residual**:
how over- or under-picked a number is relative to a uniform baseline, given engineered features
(calendar membership, "lucky" status, decade band, historical draw frequency, recency). Those
popularity estimates then bias sampling toward *under-picked* numbers, feeding the same
conditional-ROI lever as :class:`UnpopularityProvider`, but data-driven instead of prior-driven.

Design choices that keep it robust and testable:

* **GLM** (Poisson/logistic via scikit-learn) is always available.
* **Gradient boosting** uses ``xgboost`` if installed, else scikit-learn's
  ``HistGradientBoostingRegressor`` (always available) -- same role, no hard dependency.
* **Deep learning** uses a small ``torch`` MLP if installed, else it is skipped. The ensemble
  degrades gracefully and reports which members were active in diagnostics.
* The ensemble NEVER claims to predict the draw. Its docstring, name, and diagnostics all say so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..protocol import GameSpec, InferenceProvider, ProviderResult

try:  # gradient boosting: prefer xgboost, fall back to sklearn
    import xgboost as _xgb  # type: ignore

    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import torch as _torch  # type: ignore
    import torch.nn as _nn  # type: ignore

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _number_features(spec: GameSpec) -> np.ndarray:
    """Deterministic per-number feature matrix (the same features the popularity prior encodes)."""
    n = spec.main_n
    rows = []
    for num in range(1, n + 1):
        rows.append(
            [
                1.0 if num <= 31 else 0.0,      # calendar/birthday
                1.0 if num <= 12 else 0.0,      # month
                1.0 if (num == 7 or num % 7 == 0) else 0.0,  # lucky-7
                (num - 1) / (n - 1),            # normalised magnitude (low-number bias)
                float((num - 1) // 10),         # decade band
                1.0 if num % 10 == 0 else 0.0,  # round tens
            ]
        )
    return np.asarray(rows, dtype=float)


def _historical_frequency(history: pd.DataFrame, spec: GameSpec) -> np.ndarray:
    counts = np.zeros(spec.main_n, dtype=float)
    main_cols = [c for c in history.columns if str(c).lower().startswith(("ball_", "n"))][: spec.main_k]
    for c in main_cols:
        vals = pd.to_numeric(history[c], errors="coerce").dropna().astype(int)
        for v in vals:
            if 1 <= v <= spec.main_n:
                counts[v - 1] += 1.0
    total = counts.sum()
    return counts / total if total > 0 else counts


class MLEnsembleProvider(InferenceProvider):
    name = "ml_ensemble"
    description = (
        "GLM + gradient-boosting + optional deep MLP ensemble that models number crowd-popularity "
        "(NOT the draw), biasing sampling toward under-picked numbers for better conditional ROI."
    )

    def __init__(self, target: str = "popularity", random_state: int = 0) -> None:
        if target != "popularity":
            raise ValueError(
                "MLEnsembleProvider only models 'popularity'; predicting the draw is not supported "
                "because a fair draw is unlearnable."
            )
        # scikit-learn is the one hard requirement here: xgboost and torch above are genuinely
        # soft (the ensemble falls back without them), but the Ridge GLM and the sklearn boosting
        # fallback in fit() are not optional. Checking it at construction is what lets
        # ``registry.available()`` report this provider honestly -- deferring the failure to fit()
        # makes ``--all-providers`` crash mid-benchmark on a base install instead of skipping it.
        try:
            import sklearn  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ml_ensemble requires: pip install 'lottobench[ml]'"
            ) from exc
        self.random_state = int(random_state)
        self._spec: GameSpec | None = None
        self._pick_weight: np.ndarray | None = None
        self._members: list[str] = []

    def fit(self, history: pd.DataFrame, spec: GameSpec | None = None) -> "MLEnsembleProvider":
        if spec is None:
            raise ValueError("MLEnsembleProvider.fit requires a GameSpec")
        self._spec = spec
        X = _number_features(spec)
        hist_freq = _historical_frequency(history, spec)
        X = np.column_stack([X, hist_freq])
        # Target proxy: the popularity prior's per-number weight, log-scaled. In production this
        # target is replaced by real, sales-normalised pick counts (see docs/GEOGRAPHY.md).
        from ..popularity import PopularityModel

        y = np.log(PopularityModel().number_weights(spec) + 1e-9)

        preds = []
        self._members = []

        # --- GLM (always available) ---
        from sklearn.linear_model import Ridge

        glm = Ridge(alpha=1.0)
        glm.fit(X, y)
        preds.append(glm.predict(X))
        self._members.append("glm:ridge")

        # --- Gradient boosting ---
        if _HAS_XGB:
            gb = _xgb.XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05, random_state=self.random_state,
                verbosity=0,
            )
            gb.fit(X, y)
            preds.append(gb.predict(X))
            self._members.append("gbm:xgboost")
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor

            gb = HistGradientBoostingRegressor(max_depth=3, random_state=self.random_state)
            gb.fit(X, y)
            preds.append(gb.predict(X))
            self._members.append("gbm:sklearn_hist")

        # --- Deep learning (optional) ---
        if _HAS_TORCH:
            preds.append(self._fit_torch(X, y))
            self._members.append("dl:torch_mlp")

        pop_hat = np.mean(preds, axis=0)
        # Convert predicted popularity into an *inverse* sampling weight (favour under-picked).
        inv = 1.0 / (np.exp(pop_hat) + 1e-9)
        self._pick_weight = inv / inv.sum()
        return self

    def _fit_torch(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        _torch.manual_seed(self.random_state)
        Xt = _torch.tensor(X, dtype=_torch.float32)
        yt = _torch.tensor(y, dtype=_torch.float32).unsqueeze(1)
        model = _nn.Sequential(
            _nn.Linear(X.shape[1], 16), _nn.ReLU(), _nn.Linear(16, 1)
        )
        opt = _torch.optim.Adam(model.parameters(), lr=0.05)
        loss_fn = _nn.MSELoss()
        for _ in range(300):
            opt.zero_grad()
            out = model(Xt)
            loss = loss_fn(out, yt)
            loss.backward()
            opt.step()
        with _torch.no_grad():
            return model(Xt).squeeze(1).numpy()

    def propose(self, spec: GameSpec, budget: int, rng: np.random.Generator) -> ProviderResult:
        if self._pick_weight is None or self._spec is None:
            raise RuntimeError("MLEnsembleProvider.propose called before fit()")
        tickets = self._sample_distinct_tickets(
            spec, budget, rng, main_probs=self._pick_weight, star_probs=None
        )
        scores = np.array(
            [float(np.sum(np.log(self._pick_weight[[v - 1 for v in main]] + 1e-12))) for main, _ in tickets]
        )
        return ProviderResult(
            tickets=tickets, scores=scores,
            diagnostics={"members": self._members, "has_xgboost": _HAS_XGB, "has_torch": _HAS_TORCH},
        )
