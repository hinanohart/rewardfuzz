"""Base adapter: turn an arbitrary target reward function into the canonical interface.

An adapter is responsible for two things:

1. running ``target.reward_fn`` on a candidate *inside the sandbox* so side effects are captured, and
2. normalising whatever the reward function returns (``float`` / ``dict`` / :class:`RewardResult`)
   into a :class:`RewardResult` with a single comparable ``score``.
"""

from __future__ import annotations

import math
from typing import Any

from ..synth.sandbox import run_sandboxed
from ..types import AuditTarget, Candidate, CandidateKind, RewardResult

_SCORE_KEYS = ("combined_score", "score", "reward", "fitness", "value", "public")


class BaseAdapter:
    """Default adapter; subclasses override :meth:`_invoke` for ecosystem-specific calling."""

    name = "base"
    default_kind = CandidateKind.VALUE
    default_reward_max: float | None = None

    def __init__(self, target: AuditTarget, *, timeout_s: float = 5.0) -> None:
        self.target = target
        self.timeout_s = timeout_s

    # -- subclass hook -------------------------------------------------------
    def _call(self, fn: Any, payload: Any) -> Any:
        """Invoke a reward function on a payload.

        Subclasses override this to transform the payload first (e.g. the OpenEvolve adapter
        materialises program source to a file and passes the *path*). The same transformation is
        therefore applied to both the main reward function and the held-out variant.
        """

        return fn(payload)

    # -- public API ----------------------------------------------------------
    def evaluate(self, candidate: Candidate) -> RewardResult:
        raw, se = run_sandboxed(
            lambda: self._call(self.target.reward_fn, candidate.payload), timeout_s=self.timeout_s
        )
        result = self._normalise(raw, crashed=se.exception is not None or se.timed_out)
        result.wall_time_s = se.wall_time_s
        result.artifacts["side_effects"] = se.as_dict()
        if se.exception:
            result.error = se.exception
        return result

    def evaluate_held_out(self, candidate: Candidate) -> RewardResult | None:
        """Score ``candidate`` on the held-out reward variant, or ``None`` if there is none.

        Only the held-out *score* feeds the ``heldout_gap`` invariant; side-effect signals come
        from the main :meth:`evaluate` run (the held-out variant is the same function on different
        inputs, so re-capturing its side effects would be redundant). The ``crashed`` flag is
        preserved so the judge can distinguish a genuine low score from a held-out failure.
        """

        if self.target.held_out_fn is None:
            return None
        held_fn = self.target.held_out_fn
        raw, se = run_sandboxed(
            lambda: self._call(held_fn, candidate.payload),
            timeout_s=self.timeout_s,
        )
        return self._normalise(raw, crashed=se.exception is not None or se.timed_out)

    def evaluate_with_held_out(
        self, candidate: Candidate
    ) -> tuple[RewardResult, RewardResult | None]:
        """Evaluate the main reward and (if any) the held-out variant, returning both.

        The held-out score *and* its ``crashed`` flag are attached to ``result.metrics`` here, in
        one place, so every caller (the audit loop and the benchmark) feeds the structural judge a
        consistent view and the two code paths cannot drift.
        """

        result = self.evaluate(candidate)
        held = self.evaluate_held_out(candidate)
        if held is not None:
            result.metrics["held_out_score"] = held.score
            # A held-out evaluation that itself crashed is coerced to a finite score but is not
            # evidence of memorisation; the judge skips the gap test when this is set.
            result.metrics["held_out_crashed"] = held.crashed
        return result, held

    # -- normalisation -------------------------------------------------------
    def _normalise(self, raw: Any, *, crashed: bool) -> RewardResult:
        if isinstance(raw, RewardResult):
            raw.crashed = raw.crashed or crashed
            return raw
        if crashed or raw is None:
            # A reward function that errored yields the configured crash score (default 0.0).
            crash_score = float(self.target.meta.get("crash_score", 0.0))
            return RewardResult(score=crash_score, crashed=True)
        score, metrics = self._coerce(raw)
        return RewardResult(score=score, metrics=metrics)

    @staticmethod
    def _coerce(raw: Any) -> tuple[float, dict[str, Any]]:
        if isinstance(raw, bool):
            return float(raw), {}
        if isinstance(raw, (int, float)):
            return float(raw), {}
        if isinstance(raw, dict):
            for key in _SCORE_KEYS:
                if key in raw:
                    try:
                        return float(raw[key]), dict(raw)
                    except (TypeError, ValueError):
                        return math.nan, dict(raw)
            # No recognised score key: a dict without a score is itself suspicious.
            return math.nan, dict(raw)
        try:
            return float(raw), {}
        except (TypeError, ValueError):
            return math.nan, {"raw_repr": repr(raw)[:200]}
