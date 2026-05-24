"""Tests for the instrumented candidate sandbox."""

from __future__ import annotations

import os

from rewardfuzz.synth.sandbox import run_sandboxed


def test_clean_run_reports_no_side_effects():
    result, se = run_sandboxed(lambda: 1 + 1)
    assert result == 2
    assert se.is_clean()
    assert se.exception is None


def test_file_write_is_detected():
    def thunk():
        with open("artifact.txt", "w", encoding="utf-8") as fh:
            fh.write("x")
        return 1

    _result, se = run_sandboxed(thunk)
    assert "artifact.txt" in se.files_created
    assert not se.is_clean()


def test_env_mutation_is_detected_and_restored():
    key = "REWARDFUZZ_SANDBOX_PROBE"
    assert key not in os.environ

    def thunk():
        os.environ[key] = "1"
        return 1

    _result, se = run_sandboxed(thunk)
    assert key in se.env_mutated
    # The environment is restored so audits never leak between candidates.
    assert key not in os.environ


def test_sys_exit_is_captured_not_fatal():
    def thunk():
        import sys

        sys.exit(0)

    result, se = run_sandboxed(thunk)
    assert se.exit_attempted
    assert result is None


def test_sys_exit_recorded_even_if_caught_inside():
    def thunk():
        import sys

        try:
            sys.exit(0)
        except SystemExit:
            return 1.0  # mimics a buggy reward that swallows the exit
        return 0.0

    result, se = run_sandboxed(thunk)
    assert result == 1.0
    assert se.exit_attempted  # the attempt is still recorded


def test_exception_is_reported_not_propagated():
    def thunk():
        raise ValueError("boom")

    result, se = run_sandboxed(thunk)
    assert result is None
    assert se.exception is not None
    assert "ValueError" in se.exception


def test_wall_clock_timeout():
    def thunk():
        while True:
            pass

    _result, se = run_sandboxed(thunk, timeout_s=0.3)
    assert se.timed_out


def test_builtins_mutation_does_not_leak():
    """A candidate that adds to / overwrites builtins must not pollute the host process."""

    import builtins

    assert not hasattr(builtins, "__rf_probe__")
    orig_len = builtins.len

    def thunk():
        builtins.__dict__["__rf_probe__"] = 123  # add a new name (cf. the test_tamper strategy)
        builtins.len = lambda *a, **k: 0  # overwrite an existing builtin  # noqa: ARG005
        return 1

    run_sandboxed(thunk)
    assert not hasattr(builtins, "__rf_probe__")  # added name removed
    assert builtins.len is orig_len  # overwritten builtin restored


def test_overwriting_list_builtin_does_not_break_restore():
    """Restoring builtins must not depend on a builtin the candidate may have clobbered (`list`)."""

    import builtins

    orig_list = builtins.list

    def thunk():
        builtins.list = "broken"  # would break a naive `list(builtins.__dict__)` in the restorer
        return 1

    result, _se = run_sandboxed(thunk)  # must not raise out of the sandbox
    assert result == 1
    assert builtins.list is orig_list  # restored despite the restorer running mid-corruption


def test_module_global_sabotage_does_not_re_raise():
    """The never-re-raise contract must hold even if a candidate clobbers this module's globals.

    This intentionally verifies *no-escape* only. Sabotaging the auditor's own module globals is an
    in-process limitation (see the module docstring); host-state restoration is best-effort there
    and the next run_sandboxed self-heals via its entry-time cwd fallback.
    """

    from rewardfuzz.synth import sandbox as sbx

    saved = sbx.os
    cwd = os.getcwd()

    def thunk():
        sbx.os = None  # sabotage a global the cleanup path itself depends on
        return 1

    try:
        result, _se = run_sandboxed(thunk)  # cleanup will fail internally but must be contained
        assert result == 1
    finally:
        sbx.os = saved
        os.chdir(cwd)  # the sabotage blocked run_sandboxed's own cwd restore; recover it here


def test_environment_restored_even_when_original_cwd_is_deleted(tmp_path):
    """Restore must be exception-safe: a vanished cwd cannot re-raise or strand env state."""

    import shutil

    probe = "REWARDFUZZ_CWD_DEL_PROBE"
    victim = tmp_path / "victim"
    victim.mkdir()
    original = os.getcwd()
    os.chdir(victim)
    try:

        def thunk():
            os.environ[probe] = "1"
            shutil.rmtree(victim)  # delete the dir run_sandboxed will try to return to
            return 1

        # Must return normally (never re-raise) and still restore the environment.
        _result, _se = run_sandboxed(thunk)
        assert probe not in os.environ
        # run_sandboxed must leave the process in a valid cwd (not the deleted one), so the next
        # candidate's os.getcwd() cannot crash. This must hold without the test's own chdir below.
        assert os.path.isdir(os.getcwd())
    finally:
        os.chdir(original)
        os.environ.pop(probe, None)
