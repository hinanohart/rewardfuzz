# rewardfuzz autonomous build protocol (in-repo copy)

Canonical state: `.rewardfuzz-progress.json` (gitignored — holds session/chat ids, never published).
Authoritative design lives in the maintainer's memory store; this file is the in-repo pointer.

## Phases
P0 trigger → S0 name/prior-art → P1 scaffold → P2 core impl → P3 ground-truth corpus + tests + bench →
P4 CI → P5 README → P6 critic agent → P7 final review → P8 publish → P9 branch protection → P10 verify.

## Invariants (carried from the design)
- No placeholder / random benchmark numbers. README numbers are transcribed from a real `rewardfuzz bench` run.
- LLM-independent rule-based strategies are the deterministic core; CI runs without any API key.
- CI uses real asserts on metric regression thresholds — never a `grep` whitelist.
- Failures are quarantined under `experiments/_wip/`, never `rm -rf`.
- Secrets never written to the transcript, commits, or any file.
