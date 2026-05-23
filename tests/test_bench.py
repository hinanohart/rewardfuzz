"""Benchmark regression gate.

These assertions are the real CI gate (no grep whitelist): the deterministic detector must keep
perfect precision/recall on the ground-truth corpus, flag every gameable target, and never raise a
false alarm on a robust one. If the engine regresses, these fail.
"""

from __future__ import annotations

from rewardfuzz.bench import run_bench, run_detection, run_discovery
from rewardfuzz.corpus import corpus_stats


def test_corpus_is_nontrivial():
    stats = corpus_stats()
    assert stats["candidates"] >= 30
    assert stats["hack_candidates"] >= 5
    assert stats["gameable_targets"] >= 4
    assert stats["robust_targets"] >= 2


def test_detection_precision_recall_perfect():
    overall = run_detection(bootstrap_iters=200, seed=0)["overall"]
    assert overall["recall"] == 1.0, overall
    assert overall["precision"] == 1.0, overall
    assert overall["f1"] == 1.0, overall
    assert overall["fp"] == 0 and overall["fn"] == 0


def test_detection_per_domain_no_misses():
    by_domain = run_detection(bootstrap_iters=200, seed=0)["by_domain"]
    for domain, m in by_domain.items():
        assert m["recall"] == 1.0, (domain, m)
        assert m["precision"] == 1.0, (domain, m)


def test_discovery_finds_every_gameable_target():
    disc = run_discovery(seed=0, budget=8)
    assert disc["discovery_rate"] == 1.0, disc
    assert disc["discovered"] == disc["gameable_targets"]


def test_discovery_has_no_false_alarms():
    disc = run_discovery(seed=0, budget=8)
    assert disc["false_alarm_rate"] == 0.0, disc
    assert disc["mean_hackability_robust"] == 0.0


def test_gameable_clearly_separated_from_robust():
    disc = run_discovery(seed=0, budget=8)
    assert disc["mean_hackability_gameable"] > disc["mean_hackability_robust"]


def test_bench_quick_smoke():
    result = run_bench(quick=True, seed=0)
    assert result["detection"]["overall"]["f1"] == 1.0
    assert result["discovery"]["discovery_rate"] == 1.0
