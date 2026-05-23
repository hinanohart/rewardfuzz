"""Adapter for OpenEvolve / AlphaEvolve-style evaluators.

These evaluators follow the ``evaluate(program_path) -> EvaluationResult`` contract: they are
handed the *path* to a candidate program and return a metrics mapping containing a
``combined_score`` (OpenEvolve) or similar key. The candidate payload here is program source;
the adapter materialises it to ``program.py`` inside the sandbox working directory before
invoking the evaluator, so any code the evaluator runs is fully instrumented.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from ..types import CandidateKind
from .base import BaseAdapter


class OpenEvolveAdapter(BaseAdapter):
    name = "openevolve"
    default_kind = CandidateKind.PROGRAM

    def _call(self, fn: Any, payload: Any) -> Any:
        source = payload if isinstance(payload, str) else str(payload)
        # Write the program *outside* the sandbox working directory so that the adapter's own
        # file write is never mistaken for a candidate side effect; only files the candidate
        # creates (relative to cwd) are counted as side effects.
        fd, program_path = tempfile.mkstemp(suffix=".py", prefix="rf_prog_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(source)
            return fn(program_path)
        finally:
            try:
                os.unlink(program_path)
            except OSError:
                pass
