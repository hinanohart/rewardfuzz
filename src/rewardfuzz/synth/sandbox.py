"""In-process instrumented execution context for running candidate evaluations.

The reward function itself is trusted (it is the user's code); the *candidate* may be
adversarial.  ``run_sandboxed`` runs an arbitrary thunk while capturing the side-effect
signals that the structural judge relies on:

* files created in the working directory,
* environment-variable mutations,
* attempts to call ``sys.exit`` / raise ``SystemExit``,
* standard output / error,
* uncaught exceptions, and
* wall-clock timeouts (hard interrupt via ``SIGALRM`` on the main thread of POSIX systems).

This is deliberately an *in-process* sandbox: it is enough to detect the side-effect classes
that game reward functions, and it keeps the audit loop fast and dependency-free.  Full OS-level
isolation (subprocess + ``resource.setrlimit`` + network namespace) is tracked for a later
release; see the README "Threat model & limitations" section.
"""

from __future__ import annotations

import contextlib
import io
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SandboxTimeout(Exception):
    """Raised internally when a sandboxed thunk exceeds its wall-clock budget."""


@dataclass
class SideEffects:
    files_created: list[str] = field(default_factory=list)
    env_mutated: dict[str, Any] = field(default_factory=dict)
    exit_attempted: bool = False
    exit_code: Any | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    wall_time_s: float = 0.0
    exception: str | None = None

    def is_clean(self) -> bool:
        return not (self.files_created or self.env_mutated or self.exit_attempted or self.timed_out)

    def as_dict(self) -> dict[str, Any]:
        return {
            "files_created": list(self.files_created),
            "env_mutated": dict(self.env_mutated),
            "exit_attempted": self.exit_attempted,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "wall_time_s": round(self.wall_time_s, 6),
            "exception": self.exception,
            "stdout_len": len(self.stdout),
        }


def _can_use_alarm() -> bool:
    return hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()


@contextlib.contextmanager
def _alarm(timeout_s: float):
    """Hard wall-clock interrupt on POSIX main thread; best-effort no-op elsewhere."""

    if timeout_s <= 0 or not _can_use_alarm():
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise SandboxTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def run_sandboxed(
    thunk: Callable[[], Any],
    *,
    timeout_s: float = 5.0,
) -> tuple[Any, SideEffects]:
    """Run ``thunk()`` capturing side-effect telemetry.

    Returns ``(result, side_effects)``.  If the thunk raises, ``result`` is ``None`` and the
    exception text is recorded in ``side_effects.exception`` (it is *not* re-raised — a crashing
    candidate is a finding, not an error for the caller).
    """

    import sys
    import tempfile

    se = SideEffects()
    env_before = dict(os.environ)
    cwd_before = os.getcwd()
    out_buf, err_buf = io.StringIO(), io.StringIO()
    result: Any = None

    # Instrument sys.exit so an exit attempt is recorded even if the (possibly buggy) reward
    # function catches the SystemExit and returns a passing score from it.
    real_exit = sys.exit

    def _exit_wrapper(code: object = None):
        se.exit_attempted = True
        se.exit_code = code
        raise SystemExit(code)

    with tempfile.TemporaryDirectory(prefix="rewardfuzz_sbx_") as tmp:
        start = time.perf_counter()
        sys.exit = _exit_wrapper  # type: ignore[assignment]
        try:
            os.chdir(tmp)
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                with _alarm(timeout_s):
                    result = thunk()
        except SandboxTimeout:
            se.timed_out = True
        except SystemExit as exc:
            se.exit_attempted = True
            se.exit_code = exc.code
        except BaseException as exc:  # noqa: BLE001 - we report, never propagate
            se.exception = f"{type(exc).__name__}: {exc}"
        finally:
            sys.exit = real_exit
            se.wall_time_s = time.perf_counter() - start
            try:
                created = sorted(
                    str(p.relative_to(tmp)) for p in Path(tmp).rglob("*") if p.is_file()
                )
            except OSError:
                created = []
            se.files_created = created
            os.chdir(cwd_before)

    # Detect environment mutations that escaped the sandbox.
    for key, val in os.environ.items():
        if env_before.get(key) != val:
            se.env_mutated[key] = "<changed>"
    for key in env_before:
        if key not in os.environ:
            se.env_mutated[key] = "<deleted>"
    # Restore environment so audits never leak between candidates.
    os.environ.clear()
    os.environ.update(env_before)

    se.stdout = out_buf.getvalue()
    se.stderr = err_buf.getvalue()
    return result, se
