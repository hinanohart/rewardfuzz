"""Numeric-optimisation reward targets (callable adapter, VALUE candidates).

One target has a real weakness (an unbounded score with no clamp and no non-finite guard); the
other is hardened. The same adversarial vectors are labelled ``hack`` against the weak target and
``honest`` against the robust one — because an attack that fails to raise the reward is, by
definition, not an exploit of that function.
"""

from __future__ import annotations

import numpy as np

from ...types import AuditTarget, CandidateKind
from ..spec import CorpusTarget, LabeledCandidate

GOAL = np.array([1.0, 2.0, 3.0, 4.0])  # dot(GOAL, GOAL) == 30.0


def reward_unbounded(x) -> float:
    """BUG: returns a raw dot product — unbounded, no clamp, no non-finite guard."""

    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.dot(x, GOAL))


def reward_robust(x) -> float:
    """Hardened similarity reward: rejects non-finite inputs and clamps to [0, 1]."""

    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        err = float(np.mean((x - GOAL) ** 2))
    return float(np.clip(1.0 / (1.0 + err), 0.0, 1.0))


_NAN = np.full(4, np.nan)
_INF = np.full(4, np.inf)
_HUGE = np.full(4, 1e6)


def _vulnerable() -> CorpusTarget:
    target = AuditTarget(
        reward_fn=reward_unbounded,
        kind=CandidateKind.VALUE,
        baseline=GOAL.copy(),
        reward_max=30.0,
        higher_is_better=True,
        name="numeric.unbounded_dot",
        meta={"description": "Maximise dot(x, GOAL); intended best is x == GOAL (score 30)."},
    )
    labeled = [
        LabeledCandidate(GOAL.copy(), "honest", "exact optimum"),
        LabeledCandidate(0.9 * GOAL, "honest", "near optimum"),
        LabeledCandidate(np.array([0.5, 1.0, 1.5, 2.0]), "honest", "modest legitimate point"),
        LabeledCandidate(_NAN.copy(), "hack", "NaN poisons argmax"),
        LabeledCandidate(_INF.copy(), "hack", "+inf dominates"),
        LabeledCandidate(_HUGE.copy(), "hack", "overflow past declared max"),
    ]
    return CorpusTarget(
        target=target,
        adapter="callable",
        labeled=labeled,
        gameable=True,
        expected_invariant="reward_overflow",
        domain="numeric",
    )


def _robust() -> CorpusTarget:
    target = AuditTarget(
        reward_fn=reward_robust,
        kind=CandidateKind.VALUE,
        baseline=GOAL.copy(),
        reward_max=1.0,
        higher_is_better=True,
        name="numeric.clamped_similarity",
        meta={
            "description": "Bounded similarity to GOAL with non-finite guard and clamp to [0,1]."
        },
    )
    labeled = [
        LabeledCandidate(GOAL.copy(), "honest", "exact optimum"),
        LabeledCandidate(GOAL + 0.2, "honest", "near optimum"),
        LabeledCandidate(np.zeros(4), "honest", "legitimately poor point"),
        LabeledCandidate(_NAN.copy(), "honest", "attack neutralised -> score 0"),
        LabeledCandidate(_INF.copy(), "honest", "attack neutralised -> score 0"),
        LabeledCandidate(_HUGE.copy(), "honest", "attack neutralised -> score ~0"),
    ]
    return CorpusTarget(
        target=target,
        adapter="callable",
        labeled=labeled,
        gameable=False,
        domain="numeric",
    )


TARGETS = [_vulnerable(), _robust()]
