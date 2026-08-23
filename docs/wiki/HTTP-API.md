# HTTP API

A read-mostly REST layer over `lotteries_core`: **pick a provider, get a portfolio**, and inspect the
provenance behind it. It is a convenience over the library, not a product.

> **Experimental API:** responses may be inaccurate, incomplete, or based on stale third-party data.
> They are not predictions, betting recommendations, or professional advice. Do not use this API to
> make financial or wagering decisions. See [Experimental Use and Liability](Experimental-Use-and-Liability.md).

## Install and run

```bash
pip install -e ".[api]"
lotto-serve                                  # loopback:8007
lotto-serve --host 0.0.0.0 --port 8080       # explicit bind
uvicorn lotteries_core.api:app --reload      # equivalent, with auto-reload
```

Interactive schema at **http://127.0.0.1:8007/docs**, machine-readable at `/openapi.json`.

The history file comes from `LOTTERIES_HISTORY` (default `data/euromillions.csv`) and is cached for
the process's lifetime — a refresh needs a restart. That is deliberate: a portfolio's provenance
should not change silently underneath a running service.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Service identity, standing disclaimer, endpoint index |
| `GET` | `/health` | Liveness, plus whether the configured history is readable |
| `GET` | `/providers` | **The selectable strategies** |
| `GET` | `/games` | Supported game shapes and universe sizes |
| `POST` | `/portfolio` | Generate a fixed-budget portfolio from a chosen provider |
| `GET` | `/dataset` | History provenance and staleness |
| `GET` | `/ledger/{name}` | Prospective-ledger contents and standings |
| `GET` | `/openapi.json` | OpenAPI 3 schema |

## Picking a provider

`GET /providers` lists every strategy in [`lotteries_core/registry.py`](../../lotteries_core/registry.py) —
the single registry the benchmark CLI, the outcome tracker, and this API all read, so the three
cannot drift apart:

```json
[
  {"name": "frequency", "summary": "Smoothed historical-frequency weighted sampling...",
   "ablation_of": null, "optional": false, "available": true},
  {"name": "parallax_guard_ablation", "summary": "Ablation control: identical candidate pool...",
   "ablation_of": "parallax_guard", "optional": false, "available": true}
]
```

`ablation_of` makes the signal-off control discoverable next to the strategy it controls for — the
convention described in [Contributing a Provider](Contributing-a-Provider.md). `available` is `false`
when a provider needs an optional dependency that is not installed here; it is never a hard error.

Then post that name:

```bash
curl -s http://127.0.0.1:8007/portfolio \
  -H 'content-type: application/json' \
  -d '{"provider": "parallax_guard", "game": "euromillions", "budget": 3, "seed": 7}'
```

```json
{
  "provider": "parallax_guard",
  "budget": 3,
  "tickets": [
    {"main": [40, 41, 46, 47, 50], "star": [11, 12]},
    {"main": [34, 37, 39, 44, 45], "star": [3, 8]},
    {"main": [29, 33, 38, 48, 49], "star": [1, 2]}
  ],
  "metrics": {"pair_coverage": 0.0245, "expected_roi_per_ticket": -0.7142, "...": "..."},
  "diagnostics": {"mode": "guarded", "evidence_nonzero": 0, "...": "..."},
  "history": {"rows": 1972, "last_draw": "2026-08-14", "content_sha256": "857727f6..."},
  "disclaimer": "Research output only. A fair draw is unpredictable..."
}
```

Three things ride along with every portfolio deliberately:

- **`metrics`** — coverage and conditional ROI, because those are what a portfolio can actually be
  judged on. `expected_roi_per_ticket` is always negative; the game is negative-sum.
- **`history`** — the row count, newest draw, and content hash the portfolio was built from, so a
  response can be traced back to its input.
- **`disclaimer`** — in the response body, not the fine print.

The three named research versions can be selected directly in `POST /portfolio`:

| `provider` value | Version |
|---|---|
| `gingerm` | The owner's pair-co-occurrence level-set strategy |
| `claude_inference` | Claude's contrarian Perron-Frobenius inference strategy |
| `parallax` | Replication-guarded residual inference and coverage-first selection |

For example: `{"provider":"gingerm","game":"euromillions","budget":20,"seed":7}`.
`gingerm` (and its legacy name `cooccurrence_level_set`) enumerates every main combination against
every star combination, so expect a long request. The legacy technical provider names remain
available for compatibility.

## Errors

| Status | Meaning |
|---|---|
| `404` | Unknown provider, unknown game, or absent ledger/dataset |
| `422` | Schema violation (budget outside 1–200), or history unusable for that provider |
| `503` | History file missing, or a provider's optional dependency is not installed |

## What this API will not do

There is no endpoint that takes payment, places a wager, or reports a "predicted winning" ticket, and
there should never be one. `tests/test_api.py::test_no_endpoint_offers_wagering_or_payment` asserts
that no route path contains `bet`, `wager`, `stake`, `payment`, `purchase`, `checkout`, or `deposit`
— a guard on scope rather than on code, so the boundary is enforced by CI rather than by memory.

See [Project Scope](Scope.md) for the project boundary.

Any client or local app displaying a generated portfolio should display the response's `disclaimer`
prominently and unchanged. Removing it does not change the experimental status or transfer
responsibility to the maintainers.

## Testing

```bash
pytest -q tests/test_api.py     # 15 tests
```

They skip cleanly when the `api` extra is absent, so a base install still runs a green suite.
