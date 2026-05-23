"""Optional LLM ensemble judge — a tiebreaker, never an authority.

Queries one or more Hugging Face models for a structured verdict and takes a majority vote. It is
consulted only when the structural judge is *inconclusive* (high reward, no invariant fired). If
the ``[hf]`` extra or a token is missing, every method returns ``None`` and the caller falls back
to the structural verdict.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ..synth.hf_client import HFUnavailable, get_hf_client
from ..synth.prompts import JUDGE_SYSTEM, JUDGE_USER
from ..types import AttackContext, Candidate, RewardResult, Severity, Verdict
from .meta_guard import looks_injected, sanitize

NAME = "llm_ensemble"

_DEFAULT_MODELS = ("Qwen/Qwen3-32B",)
_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _parse(text: str) -> dict | None:
    match = _JSON.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _to_severity(value: object) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.MEDIUM


class LLMEnsembleJudge:
    name = NAME

    def __init__(self, models: Sequence[str] = _DEFAULT_MODELS, *, provider: str = "auto") -> None:
        self.models = tuple(models)
        self.provider = provider

    def judge(
        self, candidate: Candidate, result: RewardResult, ctx: AttackContext
    ) -> Verdict | None:
        text = candidate.payload if isinstance(candidate.payload, str) else repr(candidate.payload)
        if looks_injected(text):
            # Do not even ask the LLM; defer to the structural judge.
            return None
        prompt = JUDGE_USER.format(
            target_desc=ctx.meta.get("description", "(no description)"),
            candidate=sanitize(text),
            score=result.score,
            baseline_score=ctx.baseline_score,
        )
        votes: list[dict] = []
        for model in self.models:
            try:
                client = get_hf_client(model, provider=self.provider)
                raw = client.chat(JUDGE_SYSTEM, prompt, seed=0, temperature=0.0)
            except HFUnavailable:
                return None
            except Exception:  # noqa: BLE001 - provider/network errors -> abstain
                continue
            parsed = _parse(raw)
            if parsed is not None:
                votes.append(parsed)
        if not votes:
            return None
        hack_votes = sum(1 for v in votes if bool(v.get("is_hack")))
        is_hack = hack_votes * 2 > len(votes)
        confidence = sum(float(v.get("confidence", 0.5)) for v in votes) / len(votes)
        severities = [_to_severity(v.get("severity")) for v in votes if v.get("is_hack")]
        severity = (
            max(severities, key=lambda s: list(Severity).index(s)) if severities else Severity.NONE
        )
        rationale = next(
            (str(v.get("rationale", "")) for v in votes if v.get("is_hack")), "ensemble vote"
        )
        return Verdict(
            is_hack=is_hack,
            severity=severity if is_hack else Severity.NONE,
            confidence=round(confidence, 3),
            rationale=f"LLM ensemble ({hack_votes}/{len(votes)} hack): {rationale}",
            source=NAME,
        )
