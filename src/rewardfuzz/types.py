"""Core domain types shared across rewardfuzz.

The whole pipeline is built on four small dataclasses:

* :class:`Candidate`   — an adversarial solution produced by a strategy.
* :class:`RewardResult` — what a target reward function returns when run on a candidate,
  normalised and annotated with side-effect telemetry.
* :class:`Verdict`     — a judge's decision about whether a candidate is a hack.
* :class:`AuditTarget` — the reward function under audit plus the metadata strategies need.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class CandidateKind(str, Enum):
    """What shape a candidate payload takes, which determines how it is executed."""

    PROGRAM = "program"  # payload is Python source code to be executed
    RESPONSE = "response"  # payload is a text response (RLVR-style verification)
    VALUE = "value"  # payload is a raw input value (numeric optimisation)


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_WEIGHT: dict[Severity, float] = {
    Severity.NONE: 0.0,
    Severity.LOW: 0.25,
    Severity.MEDIUM: 0.5,
    Severity.HIGH: 0.8,
    Severity.CRITICAL: 1.0,
}


def severity_weight(sev: Severity) -> float:
    return SEVERITY_WEIGHT[sev]


@dataclass
class Candidate:
    """An adversarial solution proposed against a reward function."""

    payload: Any
    kind: CandidateKind
    origin: str  # name of the strategy that produced this candidate
    meta: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def short(self, n: int = 60) -> str:
        text = self.payload if isinstance(self.payload, str) else repr(self.payload)
        text = text.replace("\n", "\\n")
        return text if len(text) <= n else text[: n - 1] + "…"


@dataclass
class RewardResult:
    """Normalised output of running a target reward function on a candidate."""

    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    crashed: bool = False
    error: str | None = None
    wall_time_s: float = 0.0

    @property
    def side_effects(self) -> dict[str, Any]:
        return self.artifacts.get("side_effects", {})


@dataclass
class Verdict:
    """A judge's decision about a single candidate."""

    is_hack: bool
    severity: Severity
    confidence: float  # 0..1
    rationale: str
    source: str  # "structural" | "llm_ensemble" | "ensemble"
    invariant: str | None = None  # the structural invariant that fired, if any


@dataclass
class AuditTarget:
    """A reward / fitness / verifier function under audit plus what strategies need.

    ``reward_fn`` maps a *candidate payload* to either a float, a metrics ``dict`` (with a
    ``combined_score`` / ``score`` / ``reward`` key), or a :class:`RewardResult`.

    ``held_out_fn`` is an optional semantically-equivalent variant used to measure the
    visible-vs-held-out reward gap (SpecBench-style over-fitting signal).
    """

    reward_fn: Callable[[Any], Any]
    kind: CandidateKind = CandidateKind.VALUE
    baseline: Any = None  # an honest reference payload
    reward_max: float | None = None  # declared maximum legitimate reward, if known
    higher_is_better: bool = True
    held_out_fn: Callable[[Any], Any] | None = None
    name: str = "target"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackContext:
    """Everything a strategy needs to generate candidates for one target."""

    kind: CandidateKind
    baseline: Any
    reward_max: float | None
    higher_is_better: bool
    baseline_score: float | None
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Strategy(Protocol):
    """Generates adversarial candidates for a target."""

    name: str

    def propose(self, ctx: AttackContext, budget: int, rng: Any) -> Iterator[Candidate]: ...


@runtime_checkable
class Judge(Protocol):
    """Decides whether a candidate is a hack given its reward result."""

    name: str

    def judge(self, candidate: Candidate, result: RewardResult, ctx: AttackContext) -> Verdict: ...
