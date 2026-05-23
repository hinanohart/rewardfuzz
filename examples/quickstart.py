"""Minimal, self-contained rewardfuzz example.

Audits a gameable fitness function (an unbounded score with no clamp / non-finite guard) and its
hardened counterpart, printing the Hackability of each plus the top hardening suggestion.

Run with:  python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

from rewardfuzz import audit

GOAL = np.array([1.0, 2.0, 3.0, 4.0])


def gameable_fitness(x) -> float:
    # BUG: returns a raw, unbounded score with no clamp and no non-finite guard, so a NaN or an
    # enormous vector "wins" without solving anything.
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.dot(x, GOAL))


def hardened_fitness(x) -> float:
    # Rejects non-finite inputs and clamps to [0, 1].
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        err = float(np.mean((x - GOAL) ** 2))
    return float(np.clip(1.0 / (1.0 + err), 0.0, 1.0))


def main() -> None:
    for label, fn, reward_max in [
        ("gameable", gameable_fitness, 30.0),
        ("hardened", hardened_fitness, 1.0),
    ]:
        report = audit(
            fn, adapter="callable", kind="value", baseline=GOAL.copy(), reward_max=reward_max
        )
        print(f"[{label:8s}] {report.summary_line()}")
        for tip in report.hardening[:1]:
            print(f"           fix: {tip}")


if __name__ == "__main__":
    main()
