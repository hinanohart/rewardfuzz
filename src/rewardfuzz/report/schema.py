"""Machine-readable audit report.

``AuditReport`` is the public return type of :func:`rewardfuzz.audit`. It is a plain dataclass
that serialises to stable JSON so reports can be diffed in CI or stored as artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .. import __version__
from ..scoring.bootstrap import Interval
from ..scoring.robustness import ScoreSummary


def _interval_dict(iv: Interval) -> dict[str, Any]:
    return {"point": iv.point, "lo": iv.lo, "hi": iv.hi, "n": iv.n}


@dataclass
class Finding:
    strategy: str
    severity: str
    score: float
    rationale: str
    candidate_preview: str
    invariant: str | None = None
    held_out_score: float | None = None


@dataclass
class AuditReport:
    target: str
    adapter: str
    hackability: float
    robustness: float
    risk_label: str
    vulnerability: float
    n_candidates: int
    n_hacks: int
    hack_rate: dict[str, Any]
    strategies: list[dict[str, Any]]
    fired_invariants: dict[str, int]
    findings: list[Finding]
    hardening: list[str]
    config: dict[str, Any]
    rewardfuzz_version: str = __version__

    @classmethod
    def build(
        cls,
        *,
        target: str,
        adapter: str,
        summary: ScoreSummary,
        findings: list[Finding],
        hardening: list[str],
        config: dict[str, Any],
    ) -> AuditReport:
        return cls(
            target=target,
            adapter=adapter,
            hackability=summary.hackability,
            robustness=summary.robustness,
            risk_label=summary.risk_label,
            vulnerability=summary.vulnerability,
            n_candidates=summary.n_candidates,
            n_hacks=summary.n_hacks,
            hack_rate=_interval_dict(summary.hack_rate),
            strategies=[
                {
                    "name": s.name,
                    "attempts": s.attempts,
                    "hacks": s.hacks,
                    "asr": s.asr,
                    "severity_mean": s.severity_mean,
                    "channel_p": s.channel_p,
                    "top_invariant": s.top_invariant,
                }
                for s in summary.strategy_stats
            ],
            fired_invariants=summary.fired_invariants,
            findings=findings,
            hardening=hardening,
            config=config,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def summary_line(self) -> str:
        return (
            f"{self.target}: Hackability {self.hackability}/100 [{self.risk_label}] "
            f"— {self.n_hacks}/{self.n_candidates} candidates exploited"
        )
