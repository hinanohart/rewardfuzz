"""Tests for thresholds, bootstrap CIs, and the noisy-OR composite."""

from __future__ import annotations

import math

from rewardfuzz.scoring.bootstrap import bootstrap_mean
from rewardfuzz.scoring.metrics import exceeds_max, reward_is_high
from rewardfuzz.scoring.robustness import CandidateRecord, aggregate, detection_metrics
from rewardfuzz.types import AttackContext, CandidateKind, Severity, Verdict


def _ctx(reward_max=1.0, baseline=1.0, hib=True):
    return AttackContext(
        kind=CandidateKind.VALUE,
        baseline=None,
        reward_max=reward_max,
        higher_is_better=hib,
        baseline_score=baseline,
    )


def test_nan_is_always_high():
    assert reward_is_high(math.nan, _ctx()) is True


def test_inf_high_only_in_better_direction():
    assert reward_is_high(math.inf, _ctx(hib=True)) is True
    assert reward_is_high(-math.inf, _ctx(hib=True)) is False
    assert reward_is_high(-math.inf, _ctx(hib=False)) is True


def test_finite_threshold_against_reward_max():
    ctx = _ctx(reward_max=1.0, baseline=1.0)
    assert reward_is_high(0.99, ctx) is True
    assert reward_is_high(0.5, ctx) is False


def test_exceeds_max():
    ctx = _ctx(reward_max=30.0, baseline=30.0)
    assert exceeds_max(1e6, ctx) is True
    assert exceeds_max(30.0, ctx) is False


def test_bootstrap_is_deterministic_and_bracketed():
    values = [0.0, 1.0, 1.0, 0.0, 1.0]
    a = bootstrap_mean(values, iters=500, seed=0)
    b = bootstrap_mean(values, iters=500, seed=0)
    assert a.point == b.point == 0.6
    assert a.lo == b.lo and a.hi == b.hi
    assert a.lo <= a.point <= a.hi


def _rec(strategy, is_hack, severity=Severity.HIGH, invariant="x"):
    return CandidateRecord(
        strategy=strategy,
        verdict=Verdict(
            is_hack, severity if is_hack else Severity.NONE, 1.0, "", "structural", invariant
        ),
        high_reward=is_hack,
        score=1.0,
    )


def test_aggregate_noisy_or_and_labels():
    records = [_rec("degenerate", True), _rec("degenerate", False), _rec("numeric_exploit", True)]
    summary = aggregate(records, bootstrap_iters=200, seed=0)
    assert 0 < summary.hackability <= 100
    assert summary.n_hacks == 2
    assert summary.robustness < 100  # any exploit lowers robustness
    assert "x" in summary.fired_invariants


def test_aggregate_no_hacks_is_minimal_risk():
    records = [_rec("degenerate", False) for _ in range(20)]
    summary = aggregate(records, bootstrap_iters=200, seed=0)
    assert summary.hackability == 0.0
    assert summary.risk_label == "MINIMAL RISK"


def test_detection_metrics_perfect():
    truth = [True, False, True, False]
    pred = [True, False, True, False]
    m = detection_metrics(truth, pred, bootstrap_iters=100)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["accuracy"]["point"] == 1.0
