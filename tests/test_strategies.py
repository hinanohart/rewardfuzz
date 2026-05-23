"""Tests for attack strategies (kind-awareness and determinism)."""

from __future__ import annotations

import numpy as np

from rewardfuzz.strategies import DEFAULT_STRATEGIES, get_strategy, list_strategies
from rewardfuzz.types import AttackContext, CandidateKind


def ctx(kind, baseline=None, meta=None):
    return AttackContext(
        kind=kind,
        baseline=baseline,
        reward_max=1.0,
        higher_is_better=True,
        baseline_score=1.0,
        meta=meta or {"entrypoint": "solve", "param": "x"},
    )


def collect(name, c, budget=8):
    return list(get_strategy(name).propose(c, budget, np.random.default_rng(0)))


def test_default_strategies_registered():
    for name in DEFAULT_STRATEGIES:
        assert name in list_strategies()


def test_degenerate_programs():
    cands = collect("degenerate", ctx(CandidateKind.PROGRAM))
    assert cands and all(c.kind == CandidateKind.PROGRAM for c in cands)


def test_numeric_exploit_values():
    cands = collect("numeric_exploit", ctx(CandidateKind.VALUE, baseline=np.zeros(4)))
    assert cands and all(c.kind == CandidateKind.VALUE for c in cands)
    # Should include a non-finite candidate.
    assert any(not np.all(np.isfinite(np.asarray(c.payload, dtype=float))) for c in cands)


def test_test_tamper_builds_lookup():
    meta = {"entrypoint": "solve", "param": "x", "visible_io": [(1, 1), (2, 4)]}
    cands = collect("test_tamper", ctx(CandidateKind.PROGRAM, meta=meta))
    assert any("_lut" in str(c.payload) for c in cands)


def test_test_tamper_skips_value_kind():
    assert collect("test_tamper", ctx(CandidateKind.VALUE)) == []


def test_side_effect_programs_use_meta():
    meta = {"entrypoint": "solve", "param": "x", "results_file": "score.txt"}
    cands = collect("side_effect", ctx(CandidateKind.PROGRAM, meta=meta))
    assert any("score.txt" in str(c.payload) for c in cands)
    assert any("sys.exit" in str(c.payload) for c in cands)


def test_strategies_are_deterministic():
    a = [c.payload for c in collect("degenerate", ctx(CandidateKind.PROGRAM))]
    b = [c.payload for c in collect("degenerate", ctx(CandidateKind.PROGRAM))]
    assert a == b
