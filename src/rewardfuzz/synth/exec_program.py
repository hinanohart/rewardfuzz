"""Out-of-process execution of candidate programs (PROGRAM-kind candidates).

A PROGRAM candidate is *source code* — under ``--llm`` it can be authored by the LLM and is
therefore untrusted. Loading it runs its top-level code, and calling its ``solve`` entry point runs
arbitrary code on the operator's machine. To keep that code away from the operator's process (and
its secrets), this module runs each candidate in a **child** ``subprocess`` with:

* a static AST pre-scan that rejects modules used for sandbox escape / egress / introspection
  *before* the code is ever executed (:func:`scan_candidate_source`),
* a scrubbed, minimal environment — an explicit allowlist, with credential-like variables such as
  ``HF_TOKEN`` / ``OPENAI_API_KEY`` / ``AWS_*`` removed so the child cannot read them
  (:func:`scrubbed_env`), and
* best-effort POSIX resource limits in the child (CPU time + address space via
  ``resource.setrlimit``).

This is defence in depth, not a hard security boundary: there is no network namespace and no OS
container. It blocks the *in-process* secret-disclosure path the previous loader had and adds an
egress-import pre-scan, but a determined attacker on a host without further OS isolation can still
do damage. See the README "Threat model & limitations" section.

The deterministic rule-based corpus exercises the same path, so behaviour is unchanged when
``--llm`` is not set: the bundled corpus candidates contain only ``solve`` definitions and benign
side effects (file writes, ``sys.exit``) that the structural judge inspects.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import json
import os
import subprocess
import sys
from typing import Any

# Bridge so side-effect telemetry produced *in the child* (notably a top-level or in-``solve``
# ``sys.exit``) is still seen by the parent's instrumented sandbox. ``run_sandboxed`` enters
# ``collect_child_effects`` around the thunk; ``load_solve`` records what the child did into the
# active collector, and the sandbox merges it into ``SideEffects``. Files the child writes land in
# the inherited cwd (the sandbox's own temp dir) and are picked up by the sandbox's file scan, so
# only the exit signal needs this explicit channel.
_collector: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "rewardfuzz_child_effects", default=None
)


@contextlib.contextmanager
def collect_child_effects():
    """Activate a child-effect collector for the duration of one sandboxed thunk.

    Yields a dict that accumulates ``{"exit_attempted": bool, "exit_code": ...}`` from any candidate
    children run via :func:`load_solve` while it is active.
    """

    box: dict[str, Any] = {"exit_attempted": False, "exit_code": None}
    token = _collector.set(box)
    try:
        yield box
    finally:
        _collector.reset(token)


def _record_child_exit(code: Any) -> None:
    box = _collector.get()
    if box is not None and not box["exit_attempted"]:
        box["exit_attempted"] = True
        box["exit_code"] = code


# Modules whose import from candidate source is rejected outright by the pre-scan: network egress,
# process spawning, native code, parallelism, and OS/filesystem control usable to reach beyond the
# sandbox. ``sys`` is deliberately *not* here: ``sys.exit`` is one of the reward-hacking
# side-effect classes rewardfuzz is built to detect (the bundled ``EXIT_HACK`` fixture uses it),
# the exit is captured and contained, and it grants no extra reach the child does not already have.
# Secret theft via ``os.environ`` is defeated by the scrubbed child environment, not by this list.
_BANNED_IMPORTS = frozenset(
    {
        "socket",
        "subprocess",
        "os",
        "ctypes",
        "multiprocessing",
        "resource",
        "importlib",
        "shutil",
        "pty",
        "signal",
        "asyncio",
        "http",
        "urllib",
        "ftplib",
        "smtplib",
        "ssl",
    }
)

# Attribute / name tokens that indicate an attempt to break out of the loader via introspection
# (walking ``__class__``/``__subclasses__`` to reach ``os``/``subprocess``, etc.).
_BANNED_ATTRS = frozenset(
    {
        "__subclasses__",
        "__bases__",
        "__base__",
        "__mro__",
        "__globals__",
        "__builtins__",
        "__import__",
        "__loader__",
        "__getattribute__",
    }
)

# Environment variables the child is always allowed to see (locale / path / temp only). Everything
# else — and every credential-like variable in particular — is dropped.
_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PYTHONIOENCODING",
        "SYSTEMROOT",  # Windows: needed for the child interpreter to start
    }
)

# Credential-like name fragments dropped even if they somehow appear in the allowlist path.
_CREDENTIAL_TOKENS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "PASSWD", "CREDENTIAL", "API")

# Per-child wall budget headroom over the caller's timeout, and resource limits applied in the
# child where the platform supports them.
_CPU_SECONDS = 5
_ADDRESS_SPACE_BYTES = 512 * 1024 * 1024  # 512 MiB


class UnsafeCandidate(Exception):
    """Raised when the AST pre-scan rejects candidate source before execution."""


def _is_credential_like(name: str) -> bool:
    upper = name.upper()
    if upper in _ENV_ALLOWLIST:
        return False
    return any(token in upper for token in _CREDENTIAL_TOKENS) or upper.startswith("AWS_")


def scrubbed_env() -> dict[str, str]:
    """Return a minimal environment for the child: allowlist only, credentials stripped.

    Variables such as ``HF_TOKEN``, ``OPENAI_API_KEY`` and ``AWS_*`` are never forwarded, so a
    candidate cannot read them out of ``os.environ`` even if it imports nothing.
    """

    env: dict[str, str] = {}
    for key, val in os.environ.items():
        if key not in _ENV_ALLOWLIST:
            continue
        if _is_credential_like(key):
            continue
        env[key] = val
    return env


def scan_candidate_source(source: str) -> None:
    """Static pre-scan of candidate source. Raises :class:`UnsafeCandidate` if it is unsafe.

    Rejects: a syntax error, importing a banned module (``socket``/``subprocess``/``os``/``sys``/
    ``ctypes``/``multiprocessing``/…), calling ``__import__``/``eval``/``exec``/``compile``, and
    dunder attribute access used for sandbox escape (``__subclasses__``, ``__globals__``, …).
    """

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise UnsafeCandidate(f"candidate source does not parse: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _BANNED_IMPORTS:
                    raise UnsafeCandidate(f"banned import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in _BANNED_IMPORTS:
                raise UnsafeCandidate(f"banned import: from {node.module}")
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTRS:
                raise UnsafeCandidate(f"banned attribute access: {node.attr}")
        elif isinstance(node, ast.Name):
            if node.id in _BANNED_ATTRS or node.id == "__import__":
                raise UnsafeCandidate(f"banned name: {node.id}")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in {"__import__", "eval", "exec", "compile"}:
                raise UnsafeCandidate(f"banned builtin call: {fn.id}")


# The child program. Reads a JSON job from stdin (``path``, ``inputs`` and the resource ``limits``),
# applies the limits, imports the candidate, calls ``solve`` on each input, and writes a JSON result
# line per input to stdout. Run via ``python -c`` so it has no import-path dependency on rewardfuzz
# being installed in the child; the limits are passed through the job rather than templated in, so
# this stays a plain string literal with no formatting.
_CHILD = r"""
import json, sys, math, importlib.util

def _apply_limits(limits):
    try:
        import resource
    except Exception:
        return
    for res, lim in (("RLIMIT_CPU", limits["cpu"]), ("RLIMIT_AS", limits["addr"])):
        r = getattr(resource, res, None)
        if r is None:
            continue
        try:
            soft, hard = resource.getrlimit(r)
            new_hard = lim if hard in (resource.RLIM_INFINITY, -1) else min(lim, hard)
            resource.setrlimit(r, (min(lim, new_hard), new_hard))
        except (ValueError, OSError):
            pass

def main():
    job = json.load(sys.stdin)
    _apply_limits(job["limits"])
    spec = importlib.util.spec_from_file_location("rf_candidate", job["path"])
    if spec is None or spec.loader is None:
        sys.stdout.write(json.dumps({"load_error": "no spec"}) + "\n")
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit as exc:
        sys.stdout.write(json.dumps({"load_exit": exc.code if isinstance(exc.code, int) else 1}) + "\n")
        return
    except BaseException as exc:
        sys.stdout.write(json.dumps({"load_error": type(exc).__name__ + ": " + str(exc)}) + "\n")
        return
    solve = getattr(module, "solve", None)
    if solve is None:
        sys.stdout.write(json.dumps({"load_error": "no solve"}) + "\n")
        return
    for inp in job["inputs"]:
        try:
            val = solve(inp)
            if isinstance(val, bool) or isinstance(val, int):
                out = {"value": val}
            elif isinstance(val, float):
                out = {"value": val if math.isfinite(val) else None, "nonfinite": not math.isfinite(val)}
            else:
                out = {"value": val if isinstance(val, (str, list, dict, type(None))) else repr(val)}
        except SystemExit as exc:
            out = {"exit": exc.code if isinstance(exc.code, int) else 1}
        except BaseException as exc:
            out = {"error": type(exc).__name__ + ": " + str(exc)}
        sys.stdout.write(json.dumps(out) + "\n")

main()
"""


class _CaseResult:
    """One ``solve(input)`` outcome relayed from the child, replayed to grader code on demand."""

    __slots__ = ("value", "exit_code", "exited", "error", "nonfinite")

    def __init__(self, payload: dict[str, Any]) -> None:
        self.exited = "exit" in payload
        self.exit_code = payload.get("exit")
        self.error = payload.get("error")
        self.nonfinite = bool(payload.get("nonfinite"))
        self.value = float("nan") if self.nonfinite else payload.get("value")

    def replay(self) -> Any:
        """Reproduce the child's outcome in the grader's process (value / SystemExit / Exception)."""

        if self.exited:
            raise SystemExit(self.exit_code)
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.value


class SubprocessSolve:
    """A ``solve``-like callable backed by an out-of-process child.

    Grader code calls ``solve(inp)`` exactly as before; under the hood every distinct input is
    evaluated once in the hardened child and the outcome (return value, ``SystemExit``, or
    exception) is replayed here. ``None`` is returned from :func:`load_solve` when the candidate
    fails to import or is rejected by the pre-scan, matching the previous loader's contract.
    """

    def __init__(self, results: dict[Any, _CaseResult]) -> None:
        self._results = results

    def __call__(self, inp: Any) -> Any:
        case = self._results.get(inp)
        if case is None:
            # An input the child was not asked about (graders always pre-declare their cases, so
            # this is defensive): treat as a non-result rather than silently fabricating a score.
            raise RuntimeError("input not evaluated in sandbox child")
        if case.exited:
            # Record the exit attempt on the active sandbox collector *before* raising, so the
            # signal survives even when a buggy grader swallows the SystemExit (the previous
            # in-process loader recorded it the same way via the sandbox's sys.exit wrapper).
            _record_child_exit(case.exit_code)
        return case.replay()


def _run_child(program_path: str, inputs: list[Any], *, timeout_s: float) -> list[dict[str, Any]]:
    job = json.dumps(
        {
            "path": program_path,
            "inputs": inputs,
            "limits": {"cpu": _CPU_SECONDS, "addr": _ADDRESS_SPACE_BYTES},
        }
    )
    proc = subprocess.run(  # noqa: S603 - fixed interpreter + inline runner, scrubbed env
        [sys.executable, "-I", "-c", _CHILD],
        input=job,
        capture_output=True,
        text=True,
        env=scrubbed_env(),
        timeout=max(timeout_s, 0.1),
        check=False,
    )
    out: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_solve(program_path: str, inputs: list[Any], *, timeout_s: float = 5.0):
    """Pre-scan, then evaluate ``solve`` over ``inputs`` in a hardened child process.

    Returns a :class:`SubprocessSolve` callable mapping each input to its replayed outcome, or
    ``None`` if the candidate is rejected by the pre-scan, fails to import, has no ``solve``, or the
    child times out / crashes before producing a load result.
    """

    try:
        with open(program_path, encoding="utf-8") as fh:
            source = fh.read()
    except OSError:
        return None

    try:
        scan_candidate_source(source)
    except UnsafeCandidate:
        # A candidate that tries to escape the sandbox does not get to run; it scores zero, exactly
        # as an unimportable candidate does in the previous loader.
        return None

    try:
        lines = _run_child(program_path, inputs, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        return None
    except OSError:
        return None

    if not lines:
        return None
    first = lines[0]
    # A load-time exit or import error means there is no usable ``solve`` (previous loader returned
    # None for an import failure; a load-time ``sys.exit`` is surfaced via the per-input replay so
    # graders that special-case ``SystemExit`` keep their behaviour).
    if "load_error" in first:
        return None
    load_exit = first.get("load_exit") if "load_exit" in first else None
    case_lines = lines[1:] if load_exit is not None else lines
    results: dict[Any, _CaseResult] = {}
    for inp, payload in zip(inputs, case_lines, strict=False):
        results[inp] = _CaseResult(payload)
    if load_exit is not None:
        # Top-level ``sys.exit`` during import: every call replays the exit (the EXIT_HACK fixture
        # exits inside ``solve``, so this branch is for programs that exit at import time).
        exit_payload = {"exit": load_exit}
        return SubprocessSolve({inp: _CaseResult(exit_payload) for inp in inputs})
    return SubprocessSolve(results)
