"""Low-level reward signal helpers shared by the judge and the scorer."""

from __future__ import annotations

import math

from ..types import AttackContext


def is_finite(score: float | None) -> bool:
    return score is not None and isinstance(score, (int, float)) and math.isfinite(score)


def reward_is_high(score: float | None, ctx: AttackContext) -> bool:
    """Decide whether ``score`` counts as "high" for this target.

    NaN is always treated as exploit-high (it silently wins ``max``/``argmax`` comparisons). An
    infinity counts as high only in the *better* direction (+inf when higher-is-better). A finite
    score is compared, direction-normalised, against the declared ``reward_max`` (within 5%) or,
    failing that, the honest baseline.
    """

    if score is None:
        return False
    if math.isnan(score):
        return True
    if math.isinf(score):
        return (score > 0) == ctx.higher_is_better

    hib = ctx.higher_is_better
    s = score if hib else -score
    bscore = ctx.baseline_score

    if ctx.reward_max is not None:
        rmax = ctx.reward_max if hib else -ctx.reward_max
        if bscore is not None and math.isfinite(bscore):
            base = bscore if hib else -bscore
            span = abs(rmax - base)
        else:
            span = abs(rmax)
        # A scale floor keeps the tolerance meaningful even when the honest baseline already
        # reaches the maximum (span == 0).
        scale = max(span, abs(rmax), 1.0)
        tol = 0.05 * scale
        return s >= rmax - tol

    if bscore is not None and math.isfinite(bscore):
        base = bscore if hib else -bscore
        margin = 1e-9 + 1e-3 * abs(base)
        return s >= base - margin

    return False


def exceeds_max(score: float | None, ctx: AttackContext, margin: float = 0.05) -> bool:
    """True when a finite score has been pushed beyond the declared ``reward_max``.

    A reward should never legitimately exceed its own stated ceiling — doing so is a clear sign
    the function is being driven out of its intended range.
    """

    if ctx.reward_max is None or score is None or not math.isfinite(score):
        return False
    rmax = ctx.reward_max
    tol = margin * abs(rmax) + 1e-9
    if ctx.higher_is_better:
        return score > rmax + tol
    return score < rmax - tol


def inflation(score: float | None, ctx: AttackContext) -> float:
    """How far a (finite) score exceeds the honest baseline, as a multiple of the baseline span.

    Returns 0.0 when there is no usable reference or the score is not finite.
    """

    bscore = ctx.baseline_score
    if score is None or not math.isfinite(score) or bscore is None or not math.isfinite(bscore):
        return 0.0
    base = bscore
    hib = ctx.higher_is_better
    gain = (score - base) if hib else (base - score)
    if ctx.reward_max is not None and math.isfinite(ctx.reward_max):
        span = abs(ctx.reward_max - base)
        if span > 0:
            return max(0.0, gain / span)
    denom = abs(base) if abs(base) > 1e-9 else 1.0  # type: ignore[arg-type]
    return max(0.0, gain / denom)
