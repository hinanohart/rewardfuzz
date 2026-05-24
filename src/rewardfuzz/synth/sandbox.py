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

This context itself runs in-process: it is enough to detect the side-effect classes that game
reward functions, and it keeps the audit loop fast and dependency-free.  Untrusted *program*
candidates (which an LLM may author) are not run here directly — they are executed out-of-process
by ``synth.exec_program`` (subprocess + AST pre-scan + scrubbed env + ``resource.setrlimit``); the
exit signal from such a child is bridged back into the ``SideEffects`` below via
``collect_child_effects``, and any files it writes land in this context's temp dir and are picked
up by the file scan.  This is not a hard security boundary (no network namespace / container); see
the README "Threat model & limitations" section.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import os
import signal
import sys
import tempfile
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


def _restore_builtins(snapshot: dict[str, Any]) -> None:
    """Undo any mutation a candidate made to the ``builtins`` namespace.

    Drops names the candidate added and re-binds any it overwrote.  Must be called *before* any
    further use of the standard library in the cleanup path: a candidate that overwrote a builtin
    such as ``len`` would otherwise break ``glob``/``re`` (and hence the file scan below).  Uses
    only dict-view set algebra and dict methods — all resolved on the type via C slots, never
    through the namespace being repaired — so it works even if the candidate overwrote
    ``builtins.list`` (or any other builtin).
    """

    for key in builtins.__dict__.keys() - snapshot.keys():  # names the candidate added
        del builtins.__dict__[key]
    for key, val in snapshot.items():
        if builtins.__dict__.get(key) is not val:
            builtins.__dict__[key] = val


def _restore_cwd(preferred: str) -> None:
    """Return to ``preferred``; if it no longer exists, fall back to a valid directory.

    A candidate that deletes the original working directory must not strand the process in a
    non-existent cwd — the next ``os.getcwd()`` would raise and crash the audit loop.
    """

    for target in (preferred, tempfile.gettempdir(), os.sep):
        try:
            os.chdir(target)
            return
        except OSError:
            continue


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

    # Defer the import so the sandbox keeps no hard load-time dependency on the program-execution
    # bridge (and to avoid any import-order coupling between the two synth modules).
    from .exec_program import collect_child_effects

    se = SideEffects()
    env_before = dict(os.environ)
    builtins_before = dict(builtins.__dict__)
    try:
        cwd_before = os.getcwd()
    except OSError:
        # Entered while the cwd no longer exists (e.g. a prior candidate deleted it). Recover to a
        # valid directory so the audit loop self-heals rather than crashing here.
        cwd_before = tempfile.gettempdir()
        os.chdir(cwd_before)
    out_buf, err_buf = io.StringIO(), io.StringIO()
    result: Any = None

    # Instrument sys.exit so an exit attempt is recorded even if the (possibly buggy) reward
    # function catches the SystemExit and returns a passing score from it.
    real_exit = sys.exit

    def _exit_wrapper(code: object = None):
        se.exit_attempted = True
        se.exit_code = code
        raise SystemExit(code)

    try:
        # ``ignore_cleanup_errors`` keeps temp-dir teardown from raising out of the sandbox (e.g.
        # a candidate that left a background writer), preserving the never-re-raise contract.
        with tempfile.TemporaryDirectory(
            prefix="rewardfuzz_sbx_", ignore_cleanup_errors=True
        ) as tmp:
            start = time.perf_counter()
            sys.exit = _exit_wrapper  # type: ignore[assignment]
            try:
                os.chdir(tmp)
                # ``collect_child_effects`` captures side effects produced by candidate code run
                # out-of-process (e.g. a ``sys.exit`` inside a subprocess-executed PROGRAM
                # candidate) so they are merged into ``se`` below alongside the in-process signals.
                with collect_child_effects() as child_effects:
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
                # Merge any exit attempt observed in an out-of-process candidate child (the file
                # writes such a child makes land in ``tmp`` and are picked up by the scan below).
                if child_effects.get("exit_attempted") and not se.exit_attempted:
                    se.exit_attempted = True
                    se.exit_code = child_effects.get("exit_code")
                # Cleanup must never re-raise (the contract above). The whole block is guarded so
                # that even a candidate which sabotages this module's globals cannot escape; the
                # outer finally is the second line of defence.
                try:
                    # Restore builtins FIRST: the file scan below goes through glob -> re, which
                    # malfunctions if the candidate overwrote a builtin such as ``len``.
                    _restore_builtins(builtins_before)
                    try:
                        created = sorted(
                            str(p.relative_to(tmp)) for p in Path(tmp).rglob("*") if p.is_file()
                        )
                    except OSError:
                        created = []
                    se.files_created = created
                    # Leave the temp dir before TemporaryDirectory tears it down.
                    _restore_cwd(cwd_before)
                except BaseException:  # noqa: BLE001 - cleanup is best-effort, never re-raised
                    pass
    finally:
        # Always runs — even if the thunk or the temp-dir teardown raised — so a candidate can
        # never leak host state into the next audit (the isolation contract in this module's
        # docstring). Restore builtins FIRST so the rest of this block runs against a sane
        # namespace; then detect+undo env, then cwd. All steps are idempotent with the inner
        # finally (which already restored these on the normal path), and the block is guarded so
        # cleanup can never re-raise out of run_sandboxed.
        try:
            _restore_builtins(builtins_before)
            for key, val in os.environ.items():
                if env_before.get(key) != val:
                    se.env_mutated[key] = "<changed>"
            for key in env_before:
                if key not in os.environ:
                    se.env_mutated[key] = "<deleted>"
            os.environ.clear()
            os.environ.update(env_before)
            _restore_cwd(cwd_before)
        except BaseException:  # noqa: BLE001 - cleanup is best-effort, never re-raised
            pass

    se.stdout = out_buf.getvalue()
    se.stderr = err_buf.getvalue()
    return result, se
