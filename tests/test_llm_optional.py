"""The optional LLM path must degrade gracefully when unavailable."""

from __future__ import annotations

import pytest

from rewardfuzz.strategies import get_strategy
from rewardfuzz.synth.hf_client import HFUnavailable, get_hf_client
from rewardfuzz.types import AttackContext, CandidateKind


@pytest.fixture(autouse=True)
def _no_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)


def test_get_hf_client_raises_when_unavailable():
    with pytest.raises(HFUnavailable):
        get_hf_client("Qwen/Qwen3-32B")


def test_llm_strategy_yields_nothing_without_backend():
    ctx = AttackContext(
        kind=CandidateKind.RESPONSE,
        baseline=None,
        reward_max=1.0,
        higher_is_better=True,
        baseline_score=1.0,
        meta={},
    )
    cands = list(get_strategy("llm_specgaming").propose(ctx, 4, None))
    assert cands == []


def test_audit_with_llm_judge_falls_back_to_structural():
    # Requesting the LLM judge without a backend must still produce a valid structural audit.
    import numpy as np

    from rewardfuzz import audit
    from rewardfuzz.types import AuditTarget

    at = AuditTarget(
        reward_fn=lambda x: float(np.dot(np.asarray(x, dtype=float), [1, 2, 3, 4])),
        kind=CandidateKind.VALUE,
        baseline=np.array([1.0, 2.0, 3.0, 4.0]),
        reward_max=30.0,
        name="dot",
    )
    report = audit(at, adapter="callable", seed=0, judge="structural+llm")
    assert report.n_candidates > 0
