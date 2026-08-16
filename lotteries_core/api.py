"""Read-mostly HTTP API over the framework: pick a provider, get a portfolio, inspect provenance.

    pip install -e ".[api]"
    lotto-serve                      # or: uvicorn lotteries_core.api:app --reload
    open http://127.0.0.1:8000/docs  # interactive OpenAPI schema

What it exposes
---------------
``GET  /providers``  the selectable strategies, with their honest scope and their ablation controls
``POST /portfolio``  generate a fixed-budget portfolio from a chosen provider
``GET  /games``      supported game shapes
``GET  /dataset``    the history's provenance and staleness
``GET  /ledger/{name}`` prospective-ledger standings
``GET  /openapi.json`` the machine-readable schema

What it deliberately does not expose
------------------------------------
There is no endpoint that takes payment, places a wager, or reports a "predicted winning" ticket, and
there never should be. A fair draw is unpredictable; every portfolio this returns carries the same
disclaimer the rest of the repository does, in the response body rather than in the fine print. The
API is a convenience over :mod:`lotteries_core`, not a product.

The history file is read from ``LOTTERIES_HISTORY`` (default ``data/euromillions.csv``). It is loaded
once per process and cached, so a refresh needs a restart -- deliberate, since a portfolio's
provenance should not change silently underneath a running service.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised by the extras-not-installed path
    raise ImportError(
        "The HTTP API needs the 'api' extra. Install it with: pip install -e \".[api]\""
    ) from exc

from . import dataset, registry
from .coverage import coverage_report
from .popularity import PopularityModel
from .protocol import GameSpec
from .roi import JackpotModel, portfolio_expected_roi

DISCLAIMER = (
    "Research output only. A fair draw is unpredictable, the game is negative-sum, and these "
    "tickets are not more likely to win than any others. Nothing here places a wager or moves money."
)

GAMES: dict[str, GameSpec] = {
    "euromillions": GameSpec.euromillions(),
    "totoloto": GameSpec.totoloto(),
    "eurodreams": GameSpec.eurodreams(),
}

DEFAULT_HISTORY = os.environ.get("LOTTERIES_HISTORY", "data/euromillions.csv")

app = FastAPI(
    title="lotteries-core",
    version="0.1.0",
    summary="Coverage-focused lottery portfolio research framework.",
    description=__doc__,
)


# ------------------------------------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------------------------------------


class ServiceInfo(BaseModel):
    service: str
    version: str
    disclaimer: str
    history: str
    endpoints: list[str]


class ProviderInfo(BaseModel):
    name: str
    summary: str
    ablation_of: str | None = Field(
        None, description="If set, this provider is the signal-off control for the named provider."
    )
    optional: bool
    available: bool = Field(description="False when an optional dependency is missing here.")


class GameInfo(BaseModel):
    name: str
    main_n: int
    main_k: int
    star_n: int
    star_k: int
    total_tickets: int


class TicketModel(BaseModel):
    main: list[int]
    star: list[int]


class PortfolioRequest(BaseModel):
    provider: str = Field(description="A name from GET /providers.")
    game: str = Field("euromillions", description="A name from GET /games.")
    budget: int = Field(10, ge=1, le=200, description="Number of tickets to generate.")
    seed: int = Field(1234, description="Seed for providers that consume randomness.")


class PortfolioResponse(BaseModel):
    provider: str
    game: str
    budget: int
    seed: int
    tickets: list[TicketModel]
    metrics: dict
    diagnostics: dict
    history: dict
    disclaimer: str


class DatasetInfo(BaseModel):
    path: str
    rows: int
    first_draw: str
    last_draw: str
    content_sha256: str
    fetched_utc: str
    staleness_days: int
    stale: bool


class LedgerStanding(BaseModel):
    method: str
    settled_draws: int
    mean_lift: float | None
    mean_best_main: float | None


class LedgerInfo(BaseModel):
    ledger: str
    pending: int
    settled: int
    draw_keys_pending: list[str]
    standings: list[LedgerStanding]


# ------------------------------------------------------------------------------------------------
# Data access
# ------------------------------------------------------------------------------------------------


@lru_cache(maxsize=4)
def load_history(path: str) -> pd.DataFrame:
    """Load and cache a history CSV. Cached per path for the process's lifetime."""
    csv = Path(path)
    if not csv.exists():
        raise HTTPException(
            status_code=503,
            detail=f"history file {path!r} not found; run scripts/refresh_history.py",
        )
    return pd.read_csv(csv)


def _history_provenance(path: str, frame: pd.DataFrame) -> dict:
    meta = dataset.read(path)
    if meta is None:
        return {"path": path, "rows": len(frame), "metadata": "absent"}
    return {
        "path": path,
        "rows": meta.rows,
        "last_draw": meta.last_draw,
        "content_sha256": meta.content_sha256,
        "staleness_days": dataset.staleness_days(path),
    }


# ------------------------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------------------------


@app.get("/", response_model=ServiceInfo, tags=["meta"])
def root() -> ServiceInfo:
    """Service identity, the standing disclaimer, and where to go next."""
    return ServiceInfo(
        service="lotteries-core",
        version=app.version,
        disclaimer=DISCLAIMER,
        history=DEFAULT_HISTORY,
        endpoints=["/providers", "/games", "/portfolio", "/dataset", "/ledger/{name}", "/openapi.json"],
    )


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe. Reports whether the configured history is actually readable."""
    ok = Path(DEFAULT_HISTORY).exists()
    return {"status": "ok" if ok else "degraded", "history_present": ok}


@app.get("/providers", response_model=list[ProviderInfo], tags=["providers"])
def list_providers() -> list[ProviderInfo]:
    """The selectable strategies. Pass one of these names to POST /portfolio."""
    usable = set(registry.available())
    return [
        ProviderInfo(
            name=spec.name,
            summary=spec.summary,
            ablation_of=spec.ablation_of,
            optional=spec.optional,
            available=spec.name in usable,
        )
        for spec in registry.PROVIDERS.values()
    ]


@app.get("/games", response_model=list[GameInfo], tags=["providers"])
def list_games() -> list[GameInfo]:
    """Supported game shapes and the size of each combinatorial universe."""
    return [
        GameInfo(
            name=name,
            main_n=spec.main_n,
            main_k=spec.main_k,
            star_n=spec.star_n,
            star_k=spec.star_k,
            total_tickets=spec.n_tickets(),
        )
        for name, spec in GAMES.items()
    ]


@app.post("/portfolio", response_model=PortfolioResponse, tags=["providers"])
def build_portfolio(request: PortfolioRequest) -> PortfolioResponse:
    """Generate a fixed-budget portfolio from the chosen provider.

    The provider is fit on the configured history and asked for exactly ``budget`` distinct legal
    tickets. Coverage and conditional-ROI metrics are returned alongside, because those -- not any
    notion of likelihood -- are what the portfolio can actually be judged on.
    """
    if request.game not in GAMES:
        raise HTTPException(404, f"unknown game {request.game!r}; choose from {sorted(GAMES)}")
    if request.provider not in registry.PROVIDERS:
        raise HTTPException(404, f"unknown provider {request.provider!r}; see GET /providers")
    if request.provider not in registry.available():
        raise HTTPException(
            503, f"provider {request.provider!r} needs an optional dependency that is not installed"
        )

    spec = GAMES[request.game]
    frame = load_history(DEFAULT_HISTORY)
    provider = registry.create(request.provider)
    try:
        provider.fit(frame, spec)
    except TypeError:
        provider.fit(frame)
    except ValueError as exc:  # e.g. history has no rows under the game's current rules
        raise HTTPException(422, str(exc)) from exc

    result = provider.propose(spec, request.budget, np.random.default_rng(request.seed))
    for ticket in result.tickets:
        spec.validate_ticket(ticket)

    metrics = {
        **coverage_report(spec, result.tickets),
        **portfolio_expected_roi(spec, result.tickets, JackpotModel(), PopularityModel()),
    }
    return PortfolioResponse(
        provider=provider.name,
        game=request.game,
        budget=len(result.tickets),
        seed=request.seed,
        tickets=[TicketModel(main=list(m), star=list(s)) for m, s in result.tickets],
        metrics={k: float(v) for k, v in metrics.items()},
        diagnostics=result.diagnostics,
        history=_history_provenance(DEFAULT_HISTORY, frame),
        disclaimer=DISCLAIMER,
    )


@app.get("/dataset", response_model=DatasetInfo, tags=["provenance"])
def dataset_info(path: str = Query(DEFAULT_HISTORY, description="History CSV to describe")) -> DatasetInfo:
    """Provenance and staleness for a history file, from its metadata sidecar."""
    meta = dataset.read(path)
    if meta is None:
        raise HTTPException(
            404, f"no metadata for {path!r}; run scripts/refresh_history.py to generate it"
        )
    days = dataset.staleness_days(path)
    return DatasetInfo(
        path=meta.path,
        rows=meta.rows,
        first_draw=meta.first_draw,
        last_draw=meta.last_draw,
        content_sha256=meta.content_sha256,
        fetched_utc=meta.fetched_utc,
        staleness_days=days,
        stale=days > dataset.DEFAULT_STALE_AFTER_DAYS,
    )


@app.get("/ledger/{name}", response_model=LedgerInfo, tags=["provenance"])
def ledger_info(name: str) -> LedgerInfo:
    """Prospective-ledger contents and per-method standings, if any draws have settled."""
    from .outcome_tracker import PENDING, RESULTS, SETTLED, _read_jsonl

    base = Path("ledger") / name
    if not base.exists():
        raise HTTPException(404, f"no ledger at {base}")
    pending = _read_jsonl(base / PENDING)
    settled = _read_jsonl(base / SETTLED)

    standings: list[LedgerStanding] = []
    results_path = base / RESULTS
    if results_path.exists():
        frame = pd.read_csv(results_path)
        for method, sub in frame.groupby("method"):
            standings.append(
                LedgerStanding(
                    method=str(method),
                    settled_draws=int(len(sub)),
                    mean_lift=float(np.nanmean(sub["lift_mean_main"])) if len(sub) else None,
                    mean_best_main=float(np.nanmean(sub["m_best_main"])) if len(sub) else None,
                )
            )
        standings.sort(key=lambda s: (s.mean_lift is None, -(s.mean_lift or 0.0)))

    return LedgerInfo(
        ledger=name,
        pending=len(pending),
        settled=len(settled),
        draw_keys_pending=sorted({str(r["draw_key"]) for r in pending}),
        standings=standings,
    )


def main(argv: list[str] | None = None) -> None:
    """``lotto-serve`` console entry point."""
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Serve the lotteries-core HTTP API.")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback only)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="auto-reload on source changes")
    args = ap.parse_args(argv)

    print(f"[serve] history: {DEFAULT_HISTORY}  (override with LOTTERIES_HISTORY)")
    print(f"[serve] schema:  http://{args.host}:{args.port}/docs")
    uvicorn.run("lotteries_core.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
