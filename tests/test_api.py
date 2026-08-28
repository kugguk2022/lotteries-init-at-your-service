"""Tests for the HTTP API.

Skipped wholesale when the ``api`` extra is not installed, so a base install still runs a green
suite. The contract under test is the same one the rest of the framework promises: exactly `budget`
legal tickets, an honest disclaimer travelling with every portfolio, and no endpoint that could be
mistaken for a betting service.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi", reason="install the 'api' extra to test the HTTP API")
from lotteries_core import api  # noqa: E402


class ASGIClient:
    """Synchronous facade over HTTPX's native async ASGI transport."""

    def request(self, method, path, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


@pytest.fixture()
def history_csv(tmp_path):
    """A small but valid EuroMillions history, with its metadata sidecar."""
    import numpy as np

    from lotteries_core import dataset

    rng = np.random.default_rng(11)
    end = pd.Timestamp("2026-08-14")
    rows = []
    for i in range(240):
        mains = sorted(int(v) for v in rng.choice(np.arange(1, 51), 5, replace=False))
        stars = sorted(int(v) for v in rng.choice(np.arange(1, 13), 2, replace=False))
        rows.append(
            {
                "draw_date": (end - pd.Timedelta(days=4 * (239 - i))).date().isoformat(),
                **{f"ball_{j+1}": mains[j] for j in range(5)},
                **{f"star_{j+1}": stars[j] for j in range(2)},
            }
        )
    csv = tmp_path / "euromillions.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    dataset.write(csv, game="euromillions")
    return csv


@pytest.fixture()
def client(history_csv, monkeypatch):
    monkeypatch.setattr(api, "DEFAULT_HISTORY", str(history_csv))
    api.load_history.cache_clear()
    yield ASGIClient()
    api.load_history.cache_clear()


def test_root_carries_the_disclaimer(client):
    body = client.get("/").json()
    assert body["service"] == "lottobench"
    assert "unpredictable" in body["disclaimer"]
    assert "/providers" in body["endpoints"]


def test_health_reports_history_presence(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["history_present"] is True


def test_providers_are_listed_with_their_ablations(client):
    body = client.get("/providers").json()
    names = {p["name"] for p in body}
    assert {
        "gingerm",
        "spectral_contrarian",
        "parallax",
        "frequency",
        "unpopularity",
        "uniform_random",
        "garch_markov_branch",
        "sequence_transformer",
    } <= names

    by_name = {p["name"]: p for p in body}
    # Every ablation points at the provider it controls for, and that target exists.
    ablations = {n: p["ablation_of"] for n, p in by_name.items() if p["ablation_of"]}
    assert ablations, "at least one provider should ship an ablation control"
    for target in ablations.values():
        assert target in by_name

    for provider in body:
        assert provider["summary"]
    assert by_name["sequence_transformer"]["install_extra"] == "transformer"


def test_games_report_the_universe_size(client):
    body = {g["name"]: g for g in client.get("/games").json()}
    assert body["euromillions"]["main_n"] == 50
    assert body["euromillions"]["star_k"] == 2
    assert body["euromillions"]["total_tickets"] == 2118760 * 66


@pytest.mark.parametrize(
    "provider",
    ["frequency", "unpopularity", "spectral_contrarian", "parallax"],
)
def test_portfolio_returns_exactly_budget_legal_tickets(client, provider):
    response = client.post("/portfolio", json={"provider": provider, "budget": 7, "seed": 5})
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["tickets"]) == 7
    seen = set()
    for ticket in body["tickets"]:
        assert len(ticket["main"]) == 5
        assert len(set(ticket["main"])) == 5
        assert ticket["main"] == sorted(ticket["main"])
        assert all(1 <= n <= 50 for n in ticket["main"])
        assert len(ticket["star"]) == 2
        assert all(1 <= n <= 12 for n in ticket["star"])
        key = (tuple(ticket["main"]), tuple(ticket["star"]))
        assert key not in seen, "tickets must be distinct"
        seen.add(key)

    assert "pair_coverage" in body["metrics"]
    # The game is negative-sum; a positive expected ROI would mean the model is wrong.
    assert body["metrics"]["expected_roi_per_ticket"] < 0
    assert "unpredictable" in body["disclaimer"]
    assert body["history"]["content_sha256"]


def test_portfolio_is_reproducible_for_a_fixed_seed(client):
    payload = {"provider": "frequency", "budget": 5, "seed": 99}
    first = client.post("/portfolio", json=payload).json()["tickets"]
    second = client.post("/portfolio", json=payload).json()["tickets"]
    assert first == second


def test_unknown_provider_and_game_are_404(client):
    assert client.post("/portfolio", json={"provider": "nope"}).status_code == 404
    body = client.post("/portfolio", json={"provider": "nope"}).json()
    assert "GET /providers" in body["detail"]
    assert client.post(
        "/portfolio", json={"provider": "frequency", "game": "powerball"}
    ).status_code == 404


def test_budget_is_bounded_by_the_schema(client):
    assert client.post("/portfolio", json={"provider": "frequency", "budget": 0}).status_code == 422
    assert client.post("/portfolio", json={"provider": "frequency", "budget": 5000}).status_code == 422


def test_dataset_endpoint_reports_provenance_and_staleness(client, history_csv):
    body = client.get("/dataset", params={"path": str(history_csv)}).json()
    assert body["rows"] == 240
    assert body["last_draw"] == "2026-08-14"
    assert len(body["content_sha256"]) == 64
    assert isinstance(body["stale"], bool)

    missing = client.get("/dataset", params={"path": "does/not/exist.csv"})
    assert missing.status_code == 404


def test_ledger_404s_when_absent(client):
    assert client.get("/ledger/definitely-not-a-ledger").status_code == 404


def test_openapi_schema_is_served_and_well_formed(client):
    schema = client.get("/openapi.json").json()
    assert schema["openapi"].startswith("3.")
    assert schema["info"]["title"] == "LottoBench Analytics API"
    for path in (
        "/providers",
        "/games",
        "/portfolio",
        "/dataset",
        "/analytics/roi/{name}",
        "/analytics/roi/{name}/evolution",
    ):
        assert path in schema["paths"]
    assert "PortfolioRequest" in schema["components"]["schemas"]
    json.dumps(schema)  # must be serialisable for client generators


def test_realized_roi_analytics_and_evolution(client, tmp_path, monkeypatch):
    ledger = tmp_path / "ledger" / "demo"
    ledger.mkdir(parents=True)
    rows = []
    for draw_key, payout, control_payout in (
        ("2026-01-01", 0.0, 0.0),
        ("2026-01-08", 30.0, 10.0),
    ):
        stake = 10.0
        rows.append(
            {
                "schema_version": 1,
                "lottobench_version": "0.1.0a2",
                "provider_name": "frequency",
                "provider_version": "1",
                "provider_config_sha256": "config",
                "benchmark_id": "cohort",
                "game": "euromillions",
                "draw_key": draw_key,
                "currency": "GBP",
                "n_sets": 4,
                "stake": stake,
                "m_portfolio_prize": payout,
                "m_net_return": payout - stake,
                "realized_roi": (payout - stake) / stake,
                "c_portfolio_prize": control_payout,
                "c_net_return": control_payout - stake,
                "control_realized_roi": (control_payout - stake) / stake,
                "realized_roi_lift": (payout - control_payout) / stake,
                "outcome_source": "operator_verified",
                "purchase_proof_hash_present": 0,
            }
        )
    pd.DataFrame(rows).to_csv(ledger / "results.csv", index=False)
    monkeypatch.chdir(tmp_path)

    summary = client.get("/analytics/roi/demo")
    assert summary.status_code == 200, summary.text
    body = summary.json()
    assert body["records_analyzed"] == 2
    assert body["records_excluded"] == 0
    assert body["summaries"][0]["realized_roi"] == pytest.approx(0.5)
    assert body["summaries"][0]["realized_roi_lift"] == pytest.approx(1.0)

    evolution = client.get("/analytics/roi/demo/evolution").json()["points"]
    assert [point["settled_draws"] for point in evolution] == [1, 2]
    assert evolution[-1]["cumulative_realized_roi"] == pytest.approx(0.5)
    assert evolution[-1]["cumulative_roi_lift"] == pytest.approx(1.0)


def test_ledger_names_cannot_escape_ledger_directory(client):
    assert client.get("/analytics/roi/..%2Foutside").status_code in {400, 404}


def test_no_endpoint_offers_wagering_or_payment(client):
    """A guard on scope, not on code: this API must never grow a betting surface."""
    paths = " ".join(client.get("/openapi.json").json()["paths"]).lower()
    for forbidden in ("bet", "wager", "stake", "payment", "purchase", "checkout", "deposit"):
        assert forbidden not in paths
