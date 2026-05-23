# rewardfuzz — notes for AI coding assistants

rewardfuzz audits **reward / fitness / verifier functions** before they are used to train or
evolve models. It attacks a target reward function with adversarial candidate solutions and reports
how gameable it is (a `Hackability` score) plus concrete hardening suggestions.

## Layout
- `src/rewardfuzz/adapters/` — wrap a target into the canonical `RewardFunction` interface
  (`callable`, `openevolve`, `rlvr`).
- `src/rewardfuzz/strategies/` — attack generators. The four rule-based strategies
  (`degenerate`, `numeric_exploit`, `test_tamper`, `side_effect`) are deterministic and run with no
  network. `llm_specgaming` is optional and needs the `[hf]` extra + a Hugging Face token.
- `src/rewardfuzz/judge/` — decide whether a candidate is a hack. `structural` is deterministic and
  authoritative; `llm_ensemble` is an optional tiebreaker.
- `src/rewardfuzz/scoring/` — noisy-OR composite, bootstrap CIs, visible-vs-held-out gap.
- `src/rewardfuzz/corpus/` — ground-truth labelled targets used by `rewardfuzz bench`.

## Rules for changes
- Never bundle model weights; LLM access is via the user's own Hugging Face token only.
- The candidate sandbox (`synth/sandbox.py`) must stay isolated — corpus side-effect fixtures must
  not leak into the host process.
- Any number that appears in the README must come from `rewardfuzz bench`, not be hand-written.
- See `.claude/BUILD_PROTOCOL.md` for the build-state machine.
