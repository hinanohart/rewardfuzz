"""Adapter for RLVR (reinforcement learning with verifiable rewards) verifiers.

A verifier maps a model *response* to a (usually binary) reward, e.g.
``verify(response) -> {0, 1}`` with the ground truth bound in. The candidate payload is the
response text. The declared reward maximum defaults to ``1.0`` so "high reward" is judged
against a passing verification.
"""

from __future__ import annotations

from ..types import CandidateKind
from .base import BaseAdapter


class RLVRAdapter(BaseAdapter):
    name = "rlvr"
    default_kind = CandidateKind.RESPONSE
    default_reward_max = 1.0
