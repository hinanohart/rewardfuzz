"""Bootstrap confidence intervals.

Every aggregate rewardfuzz reports is accompanied by a bootstrap CI so that a number measured on
a small corpus is never mistaken for a precise constant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n: int

    def as_tuple(self) -> tuple[float, float, float]:
        return self.point, self.lo, self.hi


def bootstrap_mean(
    values: Sequence[float],
    *,
    iters: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for the mean of ``values``.

    Deterministic for a fixed ``seed``. With fewer than two samples the interval collapses to the
    point estimate (there is nothing to resample).
    """

    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        return Interval(point=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    point = float(np.mean(arr))
    if n == 1:
        return Interval(point=point, lo=point, hi=point, n=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return Interval(point=point, lo=lo, hi=hi, n=n)
