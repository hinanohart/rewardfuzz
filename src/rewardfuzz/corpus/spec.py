"""Data structures describing the ground-truth corpus.

Each :class:`CorpusTarget` pairs a reward function (an :class:`AuditTarget`) with a set of
*labelled* candidate solutions whose ground-truth status (``hack`` / ``honest``) is known. These
labels are what the detection benchmark scores precision/recall against; there are no synthetic or
random numbers anywhere in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..types import AuditTarget


@dataclass
class LabeledCandidate:
    payload: Any
    label: str  # "hack" | "honest"
    note: str = ""

    @property
    def is_hack(self) -> bool:
        return self.label == "hack"


@dataclass
class CorpusTarget:
    target: AuditTarget
    adapter: str
    labeled: list[LabeledCandidate]
    gameable: bool  # does this target have a real exploitable weakness?
    expected_invariant: str | None = None  # which invariant should catch the weakness
    domain: str = "misc"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.target.name
