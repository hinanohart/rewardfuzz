"""Judge layer: a deterministic structural judge plus an optional LLM tiebreaker.

``make_judge`` builds the judge implied by a spec string:

* ``"structural"``      — deterministic only (default; the CI-gated path).
* ``"structural+llm"``  — structural first; the LLM ensemble is consulted *only* when structural
  is inconclusive (high reward, no invariant fired).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..types import AttackContext, Candidate, RewardResult, Verdict
from .llm_ensemble import LLMEnsembleJudge
from .meta_guard import sanitize
from .structural import StructuralJudge

# Below this confidence, a non-hack structural verdict is treated as "inconclusive" and may be
# escalated to the LLM ensemble.
_INCONCLUSIVE_BELOW = 0.5


class CompositeJudge:
    name = "composite"

    def __init__(
        self,
        *,
        use_llm: bool = False,
        models: Sequence[str] = ("Qwen/Qwen3-32B",),
        provider: str = "auto",
    ) -> None:
        self.structural = StructuralJudge()
        self.use_llm = use_llm
        self.ensemble = LLMEnsembleJudge(models, provider=provider) if use_llm else None

    def judge(self, candidate: Candidate, result: RewardResult, ctx: AttackContext) -> Verdict:
        verdict = self.structural.judge(candidate, result, ctx)
        if verdict.is_hack or self.ensemble is None:
            return verdict
        if verdict.confidence >= _INCONCLUSIVE_BELOW:
            return verdict  # structurally confident that it is clean
        escalated = self.ensemble.judge(candidate, result, ctx)
        if escalated is not None:
            return escalated
        return verdict


def make_judge(spec: str = "structural", **kwargs) -> CompositeJudge:
    spec = spec.lower().strip()
    use_llm = "llm" in spec
    return CompositeJudge(use_llm=use_llm, **kwargs)


__all__ = [
    "StructuralJudge",
    "LLMEnsembleJudge",
    "CompositeJudge",
    "make_judge",
    "sanitize",
]
