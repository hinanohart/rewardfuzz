"""Scoring: thresholds, bootstrap CIs, and the Hackability / Robustness composite."""

from __future__ import annotations

from .bootstrap import Interval, bootstrap_mean
from .metrics import inflation, is_finite, reward_is_high
from .robustness import (
    CandidateRecord,
    ScoreSummary,
    StrategyStat,
    aggregate,
    detection_metrics,
)

__all__ = [
    "Interval",
    "bootstrap_mean",
    "reward_is_high",
    "is_finite",
    "inflation",
    "CandidateRecord",
    "StrategyStat",
    "ScoreSummary",
    "aggregate",
    "detection_metrics",
]
