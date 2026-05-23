"""Ground-truth benchmark — the single source of every number in the README.

Two complementary measurements, both fully deterministic and offline (no LLM, no network):

* **Detection** — run the judge over the labelled corpus candidates and score precision / recall /
  F1 against the known ``hack`` / ``honest`` labels.
* **Discovery** — run a full :func:`~rewardfuzz.audit.audit` (rule-based strategies only) on each
  target and check that gameable targets are flagged and robust ones are not.

``rewardfuzz bench`` prints these; the numbers are transcribed verbatim into the README. There are
no placeholder or random values anywhere.
"""

from __future__ import annotations

from typing import Any

from .audit import audit, prepare_runtime
from .corpus import corpus_stats, load_corpus
from .judge import make_judge
from .scoring.robustness import detection_metrics
from .types import Candidate

# A target is considered "flagged as gameable" by discovery when its Hackability reaches this.
DISCOVERY_FLAG_THRESHOLD = 40.0


def run_detection(
    *, judge: str = "structural", bootstrap_iters: int = 1000, seed: int = 0
) -> dict[str, Any]:
    judge_impl = make_judge(judge)
    truth: list[bool] = []
    predicted: list[bool] = []
    by_domain: dict[str, list[tuple[bool, bool]]] = {}

    for ct in load_corpus():
        impl, ctx = prepare_runtime(ct.target, ct.adapter)
        for cand in ct.labeled:
            candidate = Candidate(payload=cand.payload, kind=ct.target.kind, origin="corpus")
            result = impl.evaluate(candidate)
            held = impl.evaluate_held_out(candidate)
            if held is not None:
                result.metrics["held_out_score"] = held.score
            verdict = judge_impl.judge(candidate, result, ctx)
            truth.append(cand.is_hack)
            predicted.append(verdict.is_hack)
            by_domain.setdefault(ct.domain, []).append((cand.is_hack, verdict.is_hack))

    overall = detection_metrics(truth, predicted, bootstrap_iters=bootstrap_iters, seed=seed)
    domains = {
        dom: detection_metrics(
            [t for t, _ in rows], [p for _, p in rows], bootstrap_iters=bootstrap_iters, seed=seed
        )
        for dom, rows in sorted(by_domain.items())
    }
    return {"overall": overall, "by_domain": domains}


def run_discovery(*, seed: int = 0, budget: int = 8) -> dict[str, Any]:
    """Run a full audit per target.

    A gameable target counts as *discovered* when the audit finds at least one true exploit
    (``n_hacks > 0``) — the tool's job is to surface a working exploit, and finding even one means
    the function is gameable. A robust target raises a *false alarm* if any exploit is reported.
    """

    gameable = robust = found = false_alarms = 0
    game_hack: list[float] = []
    robust_hack: list[float] = []
    per_target: list[dict[str, Any]] = []
    for ct in load_corpus():
        report = audit(ct.target, adapter=ct.adapter, seed=seed, budget=budget, judge="structural")
        found_exploit = report.n_hacks > 0
        if ct.gameable:
            gameable += 1
            found += int(found_exploit)
            game_hack.append(report.hackability)
        else:
            robust += 1
            false_alarms += int(found_exploit)
            robust_hack.append(report.hackability)
        per_target.append(
            {
                "target": ct.name,
                "domain": ct.domain,
                "gameable": ct.gameable,
                "hackability": report.hackability,
                "risk_label": report.risk_label,
                "n_hacks": report.n_hacks,
                "n_candidates": report.n_candidates,
                "found_exploit": found_exploit,
                "high_risk": report.hackability >= DISCOVERY_FLAG_THRESHOLD,
                "expected_invariant": ct.expected_invariant,
            }
        )
    return {
        "gameable_targets": gameable,
        "robust_targets": robust,
        "discovered": found,
        "discovery_rate": round(found / gameable, 4) if gameable else None,
        "false_alarms": false_alarms,
        "false_alarm_rate": round(false_alarms / robust, 4) if robust else None,
        "mean_hackability_gameable": round(sum(game_hack) / len(game_hack), 1)
        if game_hack
        else None,
        "mean_hackability_robust": round(sum(robust_hack) / len(robust_hack), 1)
        if robust_hack
        else None,
        "per_target": per_target,
    }


def run_bench(*, quick: bool = False, seed: int = 0) -> dict[str, Any]:
    bootstrap_iters = 200 if quick else 1000
    budget = 6 if quick else 8
    return {
        "corpus": corpus_stats(),
        "detection": run_detection(bootstrap_iters=bootstrap_iters, seed=seed),
        "discovery": run_discovery(seed=seed, budget=budget),
        "config": {"quick": quick, "seed": seed, "judge": "structural", "llm": False},
    }
