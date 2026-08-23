"""The reproducible, provenance-carrying container exchanged between distributed nodes.

An :class:`InferenceEnvelope` is what one node writes to disk and another node reads. It is the
*only* wire format in the framework right now: distributed operation starts as **local/file
exchange** (write JSON envelopes to a shared directory), and networking is deliberately deferred
until deterministic aggregation over file envelopes is validated (see ``repurpose.md``).

Every envelope records enough to reproduce it exactly:

* the provider name and its config,
* the game spec,
* the RNG seed,
* a SHA-256 of the training data actually used,
* the git commit of the code (if available),
* the framework version and a creation timestamp,
* the proposed tickets and their provider scores.

Reproducibility is *checked*, not merely hoped for: :meth:`InferenceEnvelope.verify_data`
recomputes the data hash and :func:`data_sha256` is the single canonical hashing function.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import GameSpec, ProviderResult, Ticket

ENVELOPE_SCHEMA_VERSION = 1


def data_sha256(data: Any) -> str:
    """Canonical, order-stable SHA-256 for the training data used by a provider.

    Accepts a pandas DataFrame, a numpy array, bytes, or any JSON-serialisable object. The point
    is that two runs over byte-identical data produce the same hash, so an envelope's provenance
    can be verified independently of how the data was loaded.
    """
    if hasattr(data, "to_csv"):  # pandas DataFrame / Series
        payload = data.to_csv(index=False).encode("utf-8")
    elif isinstance(data, np.ndarray):
        payload = np.ascontiguousarray(data).tobytes()
    elif isinstance(data, (bytes, bytearray)):
        payload = bytes(data)
    else:
        payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_sha(cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:
        return None


def _ticket_to_json(t: Ticket) -> list[list[int]]:
    main, star = t
    return [list(main), list(star)]


def _ticket_from_json(obj: list[list[int]]) -> Ticket:
    main, star = obj
    return (tuple(int(x) for x in main), tuple(int(x) for x in star))


@dataclass
class InferenceEnvelope:
    """A single provider's proposal, with everything needed to reproduce and audit it."""

    provider: str
    game: GameSpec
    budget: int
    tickets: list[Ticket]
    scores: list[float]
    seed: int
    data_sha256: str
    config: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    git_sha: str | None = None
    framework_version: str = ""
    created_utc: str = ""  # ISO-8601; supplied by caller (kept explicit for determinism/tests)
    schema_version: int = ENVELOPE_SCHEMA_VERSION

    # ---- construction ----------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        *,
        provider: str,
        game: GameSpec,
        result: ProviderResult,
        seed: int,
        training_data: Any,
        config: dict | None = None,
        created_utc: str = "",
        repo_dir: str | Path | None = None,
    ) -> "InferenceEnvelope":
        """Assemble an envelope from a :class:`ProviderResult`, stamping provenance.

        ``created_utc`` is passed in rather than read from the clock so that runs are byte-for-byte
        reproducible in tests and cron/headless contexts. Callers that want a wall-clock stamp
        should format ``datetime.now(timezone.utc).isoformat()`` and pass it here.
        """
        from . import __version__

        return cls(
            provider=provider,
            game=game,
            budget=int(result_budget(result)),
            tickets=list(result.tickets),
            scores=[float(s) for s in np.asarray(result.scores, dtype=float)],
            seed=int(seed),
            data_sha256=data_sha256(training_data),
            config=dict(config or {}),
            diagnostics=dict(result.diagnostics),
            git_sha=_git_sha(repo_dir),
            framework_version=__version__,
            created_utc=created_utc,
        )

    # ---- (de)serialization -----------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["game"] = asdict(self.game)
        d["tickets"] = [_ticket_to_json(t) for t in self.tickets]
        return d

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "InferenceEnvelope":
        game = GameSpec(**d["game"])
        tickets = [_ticket_from_json(t) for t in d["tickets"]]
        known = {
            "provider", "budget", "scores", "seed", "data_sha256", "config",
            "diagnostics", "git_sha", "framework_version", "created_utc", "schema_version",
        }
        kwargs = {k: d[k] for k in known if k in d}
        return cls(game=game, tickets=tickets, **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "InferenceEnvelope":
        return cls.from_dict(json.loads(text))

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path) -> "InferenceEnvelope":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # ---- integrity -------------------------------------------------------------------------
    def verify_data(self, training_data: Any) -> bool:
        """Return True iff ``training_data`` hashes to this envelope's recorded ``data_sha256``."""
        return data_sha256(training_data) == self.data_sha256

    def validate(self) -> None:
        """Structural self-check: legal tickets, matching score length, sane budget."""
        if len(self.scores) != len(self.tickets):
            raise ValueError("scores/tickets length mismatch")
        if self.budget != len(self.tickets):
            raise ValueError("budget must equal number of tickets")
        seen: set[Ticket] = set()
        for t in self.tickets:
            self.game.validate_ticket(t)
            if t in seen:
                raise ValueError(f"duplicate ticket in envelope: {t!r}")
            seen.add(t)


def result_budget(result: ProviderResult) -> int:
    return len(result.tickets)
