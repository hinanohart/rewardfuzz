"""Rule-based hardening suggestions.

Maps the structural invariants that fired during an audit to concrete, copy-pasteable mitigations
for the reward function. These are deterministic templates — the optional LLM never auto-applies
patches; rewardfuzz is read-only with respect to the target.
"""

from __future__ import annotations

_SUGGESTIONS: dict[str, str] = {
    "nonfinite_reward": (
        "Reject non-finite rewards before they reach the optimiser: "
        "`if not math.isfinite(score): score = WORST_SCORE`. NaN silently wins argmax."
    ),
    "heldout_gap": (
        "Score on a held-out / perturbed split as well as the visible one and penalise the gap "
        "(SpecBench-style). A solution that only works on the visible cases is over-fit."
    ),
    "side_effect_exit": (
        "Run each candidate in a subprocess; treat any exit that did not complete the evaluation "
        "as a failure. Never let a caught SystemExit fall through to a passing/default reward."
    ),
    "side_effect_files": (
        "Evaluate candidates in an isolated temporary working directory and ignore (or forbid) "
        "filesystem writes. Do not read back results from files the candidate could have created."
    ),
    "side_effect_env": (
        "Snapshot and restore os.environ around each evaluation; never source any part of the "
        "reward from environment variables a candidate can set."
    ),
    "degenerate_high": (
        "Add a non-triviality gate: compare against an explicit no-op/empty baseline and require a "
        "real improvement, or measure solution complexity, before awarding high reward."
    ),
    "test_tamper": (
        "Hold out part of the grading data and keep expected outputs out of the candidate's reach; "
        "randomise or clamp inputs so a hardcoded lookup table cannot pass."
    ),
    "answer_dump": (
        "Require exactly one answer and parse it strictly; penalise responses that enumerate many "
        "candidate answers to satisfy a 'contains' check."
    ),
    "injection": (
        "Strip role/system markers from responses and never let response text modify the grading "
        "rubric. Treat the verifier prompt as untrusted-input-isolated."
    ),
}

_GENERIC = (
    "No specific invariant fired, but high-reward candidates were found. Re-run with a larger "
    "budget and the optional LLM strategy to probe further."
)


def suggest(fired_invariants: dict[str, int]) -> list[str]:
    """Return hardening tips ordered by how often each invariant fired."""

    if not fired_invariants:
        return []
    ordered = sorted(fired_invariants.items(), key=lambda kv: kv[1], reverse=True)
    # De-duplicate while preserving the count-ordered sequence.
    seen: set[str] = set()
    out: list[str] = []
    for inv, _count in ordered:
        tip = _SUGGESTIONS.get(inv, _GENERIC)
        if tip not in seen:
            seen.add(tip)
            out.append(tip)
    return out
