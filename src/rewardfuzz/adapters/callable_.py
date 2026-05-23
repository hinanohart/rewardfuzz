"""Generic adapter for any Python callable ``reward_fn(payload) -> float | dict | RewardResult``.

This is the universal fallback: the candidate payload is passed straight through. Use it for
numeric-optimisation fitness functions, custom scorers, or any reward you can express as a
function. ``kind`` defaults to ``VALUE`` but can be set on the :class:`AuditTarget`.
"""

from __future__ import annotations

from ..types import CandidateKind
from .base import BaseAdapter


class CallableAdapter(BaseAdapter):
    name = "callable"
    default_kind = CandidateKind.VALUE
