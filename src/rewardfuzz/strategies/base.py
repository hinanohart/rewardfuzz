"""Shared helpers for attack strategies.

Strategies are kind-aware generators: each yields candidates appropriate to the target's
:class:`~rewardfuzz.types.CandidateKind` and yields nothing for kinds it does not handle.
The program-kind contract used by the bundled corpus is simple: a candidate program defines a
single entry-point function (``solve`` by default) taking one parameter.
"""

from __future__ import annotations

import textwrap
from typing import Any

from ..types import AttackContext, Candidate, CandidateKind


def entrypoint_of(ctx: AttackContext) -> tuple[str, str]:
    """Return ``(function_name, param_name)`` for program candidates."""

    return ctx.meta.get("entrypoint", "solve"), ctx.meta.get("param", "x")


def make_program(ctx: AttackContext, body: str) -> str:
    """Wrap a function body into a program defining the target's entry point."""

    name, param = entrypoint_of(ctx)
    return f"def {name}({param}):\n" + textwrap.indent(textwrap.dedent(body), "    ")


def candidate(payload: Any, kind: CandidateKind, origin: str, **meta: Any) -> Candidate:
    return Candidate(payload=payload, kind=kind, origin=origin, meta=meta)
