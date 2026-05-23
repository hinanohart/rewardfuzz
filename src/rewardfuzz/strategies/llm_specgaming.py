"""Optional LLM-backed specification-gaming strategy.

Uses a Hugging Face model to synthesise creative reward-hacking candidates that the rule-based
strategies might miss. Requires the ``[hf]`` extra and a ``HF_TOKEN``; if either is missing it
yields nothing (the audit silently continues with the deterministic strategies).

This strategy is non-deterministic by nature and is therefore never part of the CI gate; it is an
exploratory amplifier on top of the deterministic core.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from ..synth.hf_client import HFUnavailable, get_hf_client
from ..synth.prompts import SPEC_GAMING_SYSTEM, SPEC_GAMING_USER
from ..types import AttackContext, Candidate, CandidateKind
from .base import candidate

NAME = "llm_specgaming"

_DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
_FENCE = re.compile(r"```[a-zA-Z0-9_]*\n(.*?)```", re.DOTALL)


def _extract(text: str) -> str:
    match = _FENCE.search(text)
    return match.group(1).strip() if match else text.strip()


class LLMSpecGamingStrategy:
    name = NAME

    def __init__(self, model: str = _DEFAULT_MODEL, *, provider: str = "auto") -> None:
        self.model = model
        self.provider = provider

    def propose(self, ctx: AttackContext, budget: int, rng: Any) -> Iterator[Candidate]:
        try:
            client = get_hf_client(self.model, provider=self.provider)
        except HFUnavailable:
            return  # no token / extra not installed — skip silently
        kind = ctx.kind
        prompt = SPEC_GAMING_USER.format(
            baseline_score=ctx.baseline_score,
            direction="higher-is-better" if ctx.higher_is_better else "lower-is-better",
            target_desc=ctx.meta.get("description", "(no description provided)"),
            kind=kind.value,
        )
        n = max(1, min(budget, 4))
        for i in range(n):
            try:
                raw = client.chat(SPEC_GAMING_SYSTEM, prompt, seed=i)
            except Exception as exc:  # noqa: BLE001 - network/provider errors are non-fatal
                ctx.meta.setdefault("llm_errors", []).append(str(exc))
                return
            payload: Any = _extract(raw)
            if kind == CandidateKind.VALUE:
                continue  # free-form LLM text is not a numeric value
            yield candidate(payload, kind, NAME, model=self.model, seed=i)
