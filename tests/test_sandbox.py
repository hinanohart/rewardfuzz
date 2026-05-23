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
