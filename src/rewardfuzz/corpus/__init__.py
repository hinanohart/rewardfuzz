"""Ground-truth corpus of reward functions with labelled candidate solutions.

``load_corpus`` returns every :class:`CorpusTarget`; the benchmark in :mod:`rewardfuzz.bench`
scores the judge against these labels. ``write_manifest`` emits a JSONL summary (the same file is
shipped as ``manifest.jsonl`` for inspection).
"""

from __future__ import annotations

import json
from pathlib import Path

from .spec import CorpusTarget, LabeledCandidate
from .targets import code, numeric, rlvr

_MODULES = (numeric, code, rlvr)


def load_corpus() -> list[CorpusTarget]:
    targets: list[CorpusTarget] = []
    for module in _MODULES:
        targets.extend(module.TARGETS)
    return targets


def corpus_stats() -> dict[str, int]:
    targets = load_corpus()
    candidates = [c for t in targets for c in t.labeled]
    return {
        "targets": len(targets),
        "gameable_targets": sum(1 for t in targets if t.gameable),
        "robust_targets": sum(1 for t in targets if not t.gameable),
        "candidates": len(candidates),
        "hack_candidates": sum(1 for c in candidates if c.is_hack),
        "honest_candidates": sum(1 for c in candidates if not c.is_hack),
    }


def write_manifest(path: str | Path) -> int:
    """Write one JSONL row per labelled candidate; returns the number of rows."""

    rows = 0
    with open(path, "w", encoding="utf-8") as fh:
        for target in load_corpus():
            for idx, cand in enumerate(target.labeled):
                payload = cand.payload
                preview = payload if isinstance(payload, str) else repr(payload)
                fh.write(
                    json.dumps(
                        {
                            "target": target.name,
                            "domain": target.domain,
                            "adapter": target.adapter,
                            "gameable_target": target.gameable,
                            "expected_invariant": target.expected_invariant,
                            "candidate_index": idx,
                            "label": cand.label,
                            "note": cand.note,
                            "preview": preview[:120],
                        }
                    )
                    + "\n"
                )
                rows += 1
    return rows


__all__ = [
    "CorpusTarget",
    "LabeledCandidate",
    "load_corpus",
    "corpus_stats",
    "write_manifest",
]
