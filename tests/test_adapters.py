"""Tests for adapters and reward-result normalisation."""

from __future__ import annotations

import pytest

from rewardfuzz.adapters import list_adapters, load_adapter
from rewardfuzz.types import AuditTarget, Candidate, CandidateKind


def _eval(adapter, target, payload, kind):
    impl = load_adapter(adapter)(target)
    return impl.evaluate(Candidate(payload=payload, kind=kind, origin="test"))


def test_callable_float():
    at = AuditTarget(reward_fn=lambda x: x * 2, kind=CandidateKind.VALUE, name="t")
    assert _eval("callable", at, 3, CandidateKind.VALUE).score == 6.0


def test_callable_dict_with_combined_score():
    at = AuditTarget(reward_fn=lambda x: {"combined_score": 0.7, "extra": 9}, name="t")
    result = _eval("callable", at, 1, CandidateKind.VALUE)
    assert result.score == 0.7
    assert result.metrics["extra"] == 9


def test_callable_crash_is_recorded():
    def boom(x):
        raise RuntimeError("nope")

    at = AuditTarget(reward_fn=boom, name="t")
    result = _eval("callable", at, 1, CandidateKind.VALUE)
    assert result.crashed
    assert result.score == 0.0


def test_openevolve_writes_program_and_runs():
    def reward(program_path):
        with open(program_path, encoding="utf-8") as fh:
            return 1.0 if "solve" in fh.read() else 0.0

    at = AuditTarget(reward_fn=reward, kind=CandidateKind.PROGRAM, name="t")
    result = _eval("openevolve", at, "def solve(x):\n    return x\n", CandidateKind.PROGRAM)
    assert result.score == 1.0


def test_rlvr_defaults_and_eval():
    assert load_adapter("rlvr").default_reward_max == 1.0
    at = AuditTarget(
        reward_fn=lambda r: 1.0 if "42" in r else 0.0, kind=CandidateKind.RESPONSE, name="t"
    )
    assert _eval("rlvr", at, "the answer is 42", CandidateKind.RESPONSE).score == 1.0


def test_held_out_evaluation():
    at = AuditTarget(reward_fn=lambda x: 1.0, held_out_fn=lambda x: 0.0, name="t")
    impl = load_adapter("callable")(at)
    cand = Candidate(payload=1, kind=CandidateKind.VALUE, origin="test")
    assert impl.evaluate_held_out(cand).score == 0.0

    at2 = AuditTarget(reward_fn=lambda x: 1.0, name="t2")
    impl2 = load_adapter("callable")(at2)
    assert impl2.evaluate_held_out(cand) is None


def test_unknown_adapter_raises():
    with pytest.raises(KeyError):
        load_adapter("does-not-exist")


def test_builtin_adapters_registered():
    for name in ("callable", "openevolve", "rlvr"):
        assert name in list_adapters()
