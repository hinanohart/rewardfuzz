# rewardfuzz

> **Attack your reward function before it attacks your training run.**
> A preventive, *a-priori* auditor for reward / fitness / verifier functions.

[![CI](https://github.com/hinanohart/rewardfuzz/actions/workflows/ci.yml/badge.svg)](https://github.com/hinanohart/rewardfuzz/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

rewardfuzz takes a reward / fitness / verifier function — an OpenEvolve evaluator, an RLVR
verifier, a fitness function for an evolutionary search, or any Python callable — and **attacks it
with adversarial candidate solutions before you ever train or evolve against it.** It reports a
**Hackability** score (0–100, higher = more gameable), the specific exploits it found, and concrete
hardening suggestions.

The point is *prevention*. Reward hacking is now well studied, but the tooling is almost entirely
**post-hoc detection** — watching training runs or agent trajectories for hacking after the fact.
rewardfuzz moves the check left: spend 30 seconds fuzzing the reward function so you don't spend an
hour training a model that learns to game it.

```text
[gameable] gameable_fitness: Hackability 75.0/100 [HIGH RISK]    — 3/6 candidates exploited
           fix: Reject non-finite rewards before they reach the optimiser:
                `if not math.isfinite(score): score = WORST_SCORE`. NaN silently wins argmax.
[hardened] hardened_fitness: Hackability  0.0/100 [MINIMAL RISK] — 0/6 candidates exploited
```

*(verbatim output of [`examples/quickstart.py`](examples/quickstart.py))*

---

## Install

Not yet on PyPI — install from source:

```bash
git clone https://github.com/hinanohart/rewardfuzz
cd rewardfuzz
pip install -e .              # core, fully offline, no API keys
pip install -e ".[hf]"        # + optional LLM-backed strategy/judge (needs a Hugging Face token)
```

Requires Python ≥ 3.10. Core dependencies are just `numpy` and `rich`.

## Quickstart (Python)

```python
import numpy as np
from rewardfuzz import audit

GOAL = np.array([1.0, 2.0, 3.0, 4.0])

def fitness(x):                       # an unbounded score with no clamp / non-finite guard
    return float(np.dot(np.asarray(x, float), GOAL))

report = audit(fitness, adapter="callable", kind="value",
               baseline=GOAL.copy(), reward_max=30.0)

print(report.hackability)             # 75.0
print(report.risk_label)              # "HIGH RISK"
for finding in report.findings:
    print(finding.invariant, finding.rationale)
for tip in report.hardening:
    print(tip)
```

`audit(...)` is deterministic for a fixed `seed` and needs no network with the default
`judge="structural"`.

## Quickstart (CLI)

```bash
rewardfuzz bench                       # run the ground-truth benchmark (numbers below)
rewardfuzz audit mymod:myreward --adapter openevolve --reward-max 1.0 -o report.json
rewardfuzz report report.json --md     # render a saved report as Markdown
rewardfuzz harden report.json          # just the hardening suggestions
rewardfuzz list-strategies
```

## How it works

```
target reward fn ──▶ adapter ──▶ strategies generate candidates ──▶ sandboxed eval ──▶ judge ──▶ score
```

- **Adapters** wrap a target into one interface: `callable` (any Python function), `openevolve`
  (`evaluate(program_path) -> metrics`), and `rlvr` (`verify(response) -> {0,1}`).
- **Strategies** generate adversarial candidates. The four rule-based strategies are deterministic
  and need no network:
  - `degenerate` — empty / constant / identity solutions,
  - `numeric_exploit` — NaN / ±inf / overflow / values past the declared maximum,
  - `test_tamper` — hardcoded lookups of the visible grading data, weakened builtins,
  - `side_effect` — writing a results file the grader trusts, mutating env, `sys.exit()`.
  - `llm_specgaming` *(optional, `[hf]`)* — an LLM proposes creative spec-gaming candidates.
- **Judge.** A deterministic **structural judge** is the authority — it checks a fixed set of
  invariants (`nonfinite_reward`, `reward_overflow`, `heldout_gap`, `side_effect_*`,
  `degenerate_high`, `test_tamper`, `answer_dump`, `injection`) and cannot itself be gamed. An
  optional LLM ensemble is consulted *only* to break ties the structural judge leaves open.
- **Scoring.** A noisy-OR composite over strategies (any one working exploit drives Hackability
  up), with bootstrap confidence intervals and a visible-vs-held-out gap signal.

## Benchmark

Every number here is produced by `rewardfuzz bench` on the bundled **ground-truth corpus** and is
reproduced in CI. It is deterministic and runs with no LLM and no network. This is a regression /
sanity benchmark on rewardfuzz's *own* labelled corpus — not an independent third-party evaluation.

Corpus: **8 reward functions** (5 deliberately gameable, 3 hardened), **59 labelled candidates**
(16 hack, 43 honest) across the numeric, code, and RLVR domains.

**Detection** — the structural judge vs the ground-truth labels:

| metric | value |
|---|---|
| precision | **1.00** |
| recall | **1.00** |
| F1 | **1.00** |
| confusion | tp=16, fp=0, fn=0, tn=43 |

Per domain: numeric (n=23), code (n=22), rlvr (n=14) — all precision = recall = 1.00.

**Discovery** — a full audit per target:

- gameable targets flagged with ≥1 true exploit: **5 / 5**
- false alarms on hardened targets: **0 / 3**
- mean Hackability: **gameable 55.7** vs **robust 0.0**

| target | Hackability | verdict |
|---|--:|---|
| numeric.unbounded_dot | 75.0 | HIGH RISK |
| code.catches_exit | 65.0 | MEDIUM RISK |
| code.reads_results_file | 62.7 | MEDIUM RISK |
| code.visible_only | 53.3 | MEDIUM RISK |
| rlvr.contains_substring | 22.6 | LOW RISK |
| *(3 hardened targets)* | 0.0 | MINIMAL RISK |

65 tests, 87% line coverage. (CI runs `rewardfuzz bench --quick`, which uses a smaller per-strategy
budget, so a budget-sensitive target's exact Hackability can differ by ~1 point from the full run
above; precision/recall and the discovery rate are unaffected.)

## Optional LLM features (experimental)

The `[hf]` extra enables an LLM-backed strategy (`llm_specgaming`) and an ensemble judge via the
Hugging Face Inference API. They are **experimental amplifiers on top of the deterministic core**:
non-deterministic, never part of the CI gate, and they degrade silently to a no-op when no
`HF_TOKEN` is set. Model weights are never bundled — access is only through your own token. Default
models: `Qwen/Qwen2.5-Coder-32B-Instruct` (synthesis), `Qwen/Qwen3-32B` (judge).

## Threat model & limitations

- **In-process sandbox.** Candidate evaluations run in an instrumented in-process sandbox that
  captures file writes, env mutations, `sys.exit`, exceptions, and wall-clock timeouts (hard
  `SIGALRM` interrupt on POSIX). This is enough to detect the side-effect classes that game reward
  functions, but it is **not** a security boundary against deliberately malicious code. Only audit
  reward functions and candidates you trust. Full OS-level isolation (subprocess + `setrlimit` +
  network namespace) is on the roadmap.
- The `degenerate_high` heuristic assumes the task requires real computation; a task whose correct
  answer genuinely is a constant could be mis-flagged.
- The bundled corpus is small and is rewardfuzz's own ground truth, so the perfect detection score
  reflects internal consistency, not a third-party benchmark. Treat Hackability as a comparative
  signal between reward functions, not an absolute guarantee.

## How this differs from prior work

- **Detection tools** (runtime reward-hack monitors, trajectory classifiers, coding-task hack
  benchmarks) act *during or after* training. rewardfuzz acts *before*, on the reward function
  itself, as a developer tool.
- **"Adversarial Reward Auditing" (ARA, arXiv:2602.01750)** is a *training-time RLHF method* that
  learns a Hacker/Auditor game and gates reward signals during training. rewardfuzz is the
  opposite: a standalone, pre-deployment, read-only linter/fuzzer for an arbitrary reward function.
  It changes nothing about your training loop and proposes hardening you apply yourself.

## Roadmap

- Subprocess + `setrlimit` OS-level sandbox (`[fuzz]` extra).
- Coverage-guided (Atheris) and property-based (Hypothesis) strategies.
- `shinka` adapter (ShinkaEvolve `evaluate.py`) and a generic `cli` adapter.
- A larger, domain-diverse corpus.
- Calibrated robustness scoring (`corpus/calibration.json`); today the shrink factor is a fixed
  formula rather than calibrated against the corpus.

## License

Apache-2.0. See [LICENSE](LICENSE).
