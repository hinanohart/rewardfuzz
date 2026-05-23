"""End-to-end audit tests on hand-built vulnerable and robust targets."""

from __future__ import annotations

import numpy as np

from rewardfuzz import audit
from rewardfuzz.types import AuditTarget, CandidateKind

GOAL = np.array([1.0, 2.0, 3.0, 4.0])


def _unbounded(x):
    return float(np.dot(np.asarray(x, dtype=float), GOAL))


def _robust(x):
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        return 0.0
    return float(np.clip(1.0 / (1.0 + np.mean((x - GOAL) ** 2)), 0.0, 1.0))


def test_audit_flags_vulnerable_reward():
    at = AuditTarget(
        reward_fn=_unbounded,
        kind=CandidateKind.VALUE,
        baseline=GOAL.copy(),
        reward_max=30.0,
        name="dot",
    )
    report = audit(at, adapter="callable", seed=0)
    assert report.n_hacks > 0
    assert report.hackability > 0
    assert report.hardening  # suggestions present


def test_audit_clean_on_robust_reward():
    at = AuditTarget(
        reward_fn=_robust,
        kind=CandidateKind.VALUE,
        baseline=GOAL.copy(),
        reward_max=1.0,
        name="robust",
    )
    report = audit(at, adapter="callable", seed=0)
    assert report.n_hacks == 0
    assert report.hackability == 0.0
    assert report.risk_label == "MINIMAL RISK"


def test_audit_is_deterministic():
    at = AuditTarget(
        reward_fn=_unbounded,
        kind=CandidateKind.VALUE,
        baseline=GOAL.copy(),
        reward_max=30.0,
        name="dot",
    )
    a = audit(at, adapter="callable", seed=0)
    b = audit(at, adapter="callable", seed=0)
    assert a.hackability == b.hackability
    assert a.to_json() == b.to_json()


def test_audit_report_json_is_valid():
    import json

    at = AuditTarget(
        reward_fn=_unbounded,
        kind=CandidateKind.VALUE,
        baseline=GOAL.copy(),
        reward_max=30.0,
        name="dot",
    )
    report = audit(at, adapter="callable", seed=0)
    data = json.loads(report.to_json())
    assert data["target"] == "dot"
    assert data["rewardfuzz_version"]
    assert "strategies" in data and data["strategies"]


def test_audit_accepts_plain_callable():
    report = audit(
        _unbounded, adapter="callable", baseline=GOAL.copy(), reward_max=30.0, kind="value"
    )
    assert report.n_candidates > 0
