"""The audit orchestrator — the public entry point :func:`audit`.

Pipeline: wrap the target in an adapter → measure the honest baseline → run each strategy to
generate adversarial candidates → run the (optionally held-out) reward function on each in the
sandbox → judge each candidate → aggregate into a :class:`~rewardfuzz.report.schema.AuditReport`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .adapters import load_adapter
from .judge import make_judge
from .report.hardening import suggest
from .report.schema import AuditReport, Finding
from .scoring.metrics import reward_is_high
from .scoring.robustness import CandidateRecord, aggregate
from .strategies import DEFAULT_STRATEGIES, get_strategy
from .strategies.llm_specgaming import LLMSpecGamingStrategy
from .synth.prompts import PROMPT_VERSION
from .types import AttackContext, AuditTarget, Candidate, CandidateKind

_DEFAULT_SYNTH_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
_DEFAULT_JUDGE_MODELS: tuple[str, ...] = ("Qwen/Qwen3-32B",)


def _as_target(target: Any, adapter_name: str, **kw: Any) -> AuditTarget:
    if isinstance(target, AuditTarget):
        return target
    if not callable(target):
        raise TypeError("target must be a callable reward function or an AuditTarget")
    adapter_cls = load_adapter(adapter_name)
    kind = kw.get("kind") or adapter_cls.default_kind
    if isinstance(kind, str):
        kind = CandidateKind(kind)
    meta = dict(kw.get("meta") or {})
    if kw.get("description"):
        meta.setdefault("description", kw["description"])
    return AuditTarget(
        reward_fn=target,
        kind=kind,
        baseline=kw.get("baseline"),
        reward_max=kw.get("reward_max"),
        higher_is_better=kw.get("higher_is_better", True),
        held_out_fn=kw.get("held_out"),
        name=kw.get("name", getattr(target, "__name__", "target")),
        meta=meta,
    )


def _make_strategy(name: str, synth_model: str, provider: str):
    if name == "llm_specgaming":
        return LLMSpecGamingStrategy(synth_model, provider=provider)
    return get_strategy(name)


def prepare_runtime(at: AuditTarget, adapter_name: str, *, timeout_s: float = 5.0):
    """Instantiate the adapter and build the :class:`AttackContext` for a target.

    Returns ``(adapter_impl, ctx)``. Shared by :func:`audit` and the benchmark so the baseline and
    threshold logic can never drift between them.
    """

    adapter_cls = load_adapter(adapter_name)
    impl = adapter_cls(at, timeout_s=timeout_s)
    reward_max = at.reward_max if at.reward_max is not None else adapter_cls.default_reward_max
    baseline_score: float | None = None
    if at.baseline is not None:
        baseline_score = impl.evaluate(
            Candidate(payload=at.baseline, kind=at.kind, origin="baseline")
        ).score
    ctx = AttackContext(
        kind=at.kind,
        baseline=at.baseline,
        reward_max=reward_max,
        higher_is_better=at.higher_is_better,
        baseline_score=baseline_score,
        meta=at.meta,
    )
    return impl, ctx


def audit(
    target: Callable[[Any], Any] | AuditTarget,
    *,
    adapter: str = "callable",
    kind: CandidateKind | str | None = None,
    baseline: Any = None,
    reward_max: float | None = None,
    higher_is_better: bool = True,
    held_out: Callable[[Any], Any] | None = None,
    description: str | None = None,
    meta: dict[str, Any] | None = None,
    strategies: Sequence[str] | None = None,
    seed: int = 0,
    budget: int = 8,
    judge: str = "structural",
    hf_model_synth: str = _DEFAULT_SYNTH_MODEL,
    hf_models_judge: Sequence[str] = _DEFAULT_JUDGE_MODELS,
    hf_provider: str = "auto",
    bootstrap_iters: int = 1000,
    timeout_s: float = 5.0,
) -> AuditReport:
    """Audit a reward / fitness / verifier function and return an :class:`AuditReport`.

    Parameters mirror the design doc; the only required argument is ``target``. With the default
    ``judge="structural"`` and the rule-based strategies, the whole run is deterministic for a
    fixed ``seed`` and needs no network access.
    """

    at = _as_target(
        target,
        adapter,
        kind=kind,
        baseline=baseline,
        reward_max=reward_max,
        higher_is_better=higher_is_better,
        held_out=held_out,
        description=description,
        meta=meta,
    )
    impl, ctx = prepare_runtime(at, adapter, timeout_s=timeout_s)
    effective_reward_max = ctx.reward_max
    baseline_score = ctx.baseline_score

    strat_names = list(strategies) if strategies is not None else list(DEFAULT_STRATEGIES)
    if "llm" in judge.lower() and "llm_specgaming" not in strat_names:
        strat_names.append("llm_specgaming")

    judge_impl = make_judge(judge, models=tuple(hf_models_judge), provider=hf_provider)

    records: list[CandidateRecord] = []
    findings: list[Finding] = []
    for s_idx, name in enumerate(strat_names):
        strategy = _make_strategy(name, hf_model_synth, hf_provider)
        # Independent stream per (seed, strategy): an array seed avoids the overlap of
        # ``default_rng(seed + s_idx)`` (where seed=0/s_idx=1 and seed=1/s_idx=0 collide) once a
        # strategy actually consumes ``rng``. The rule-based strategies currently do not.
        rng = np.random.default_rng([seed, s_idx])
        for candidate in strategy.propose(ctx, budget, rng):
            result, held = impl.evaluate_with_held_out(candidate)
            verdict = judge_impl.judge(candidate, result, ctx)
            records.append(
                CandidateRecord(
                    strategy=name,
                    verdict=verdict,
                    high_reward=reward_is_high(result.score, ctx),
                    score=result.score,
                    held_out_score=held.score if held is not None else None,
                )
            )
            if verdict.is_hack:
                findings.append(
                    Finding(
                        strategy=name,
                        severity=verdict.severity.value,
                        score=result.score,
                        rationale=verdict.rationale,
                        candidate_preview=candidate.short(),
                        invariant=verdict.invariant,
                        held_out_score=held.score if held is not None else None,
                    )
                )

    summary = aggregate(records, bootstrap_iters=bootstrap_iters, seed=seed)
    findings.sort(key=lambda f: f.score, reverse=at.higher_is_better)
    config = {
        "seed": seed,
        "budget": budget,
        "strategies": strat_names,
        "judge": judge,
        "adapter": adapter,
        "kind": at.kind.value,
        "reward_max": effective_reward_max,
        "higher_is_better": at.higher_is_better,
        "baseline_score": baseline_score,
        "bootstrap_iters": bootstrap_iters,
        "prompt_version": PROMPT_VERSION,
    }
    return AuditReport.build(
        target=at.name,
        adapter=adapter,
        summary=summary,
        findings=findings,
        hardening=suggest(summary.fired_invariants),
        config=config,
    )
