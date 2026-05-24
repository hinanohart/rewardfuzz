"""Tests for the out-of-process candidate program executor (synth.exec_program).

These exercise the hardened PROGRAM-candidate path with benign, inert inputs only: no LLM call, no
network, and no genuinely malicious code is ever run — the rejection cases are caught by the static
pre-scan *before* any execution, and the execution cases run trivial arithmetic in a child process.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from rewardfuzz.synth.exec_program import (
    UnsafeCandidate,
    load_solve,
    scan_candidate_source,
    scrubbed_env,
)


def _write_program(source: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py", prefix="rf_test_prog_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(source)
    return path


# -- AST pre-scan ----------------------------------------------------------------------------
def test_prescan_rejects_import_socket():
    with pytest.raises(UnsafeCandidate):
        scan_candidate_source("import socket\ndef solve(x):\n    return x\n")


def test_prescan_rejects_import_os():
    with pytest.raises(UnsafeCandidate):
        scan_candidate_source("import os\ndef solve(x):\n    return os.getpid()\n")


def test_prescan_rejects_from_import_subprocess():
    with pytest.raises(UnsafeCandidate):
        scan_candidate_source("from subprocess import run\ndef solve(x):\n    return x\n")


def test_prescan_rejects_dunder_subclasses_escape():
    src = "def solve(x):\n    return ().__class__.__bases__[0].__subclasses__()\n"
    with pytest.raises(UnsafeCandidate):
        scan_candidate_source(src)


def test_prescan_rejects_eval_call():
    with pytest.raises(UnsafeCandidate):
        scan_candidate_source("def solve(x):\n    return eval('x')\n")


def test_prescan_allows_benign_program():
    # Must not raise: the honest corpus solution uses only arithmetic.
    scan_candidate_source("def solve(x):\n    return x * x\n")


def test_prescan_allows_sys_exit():
    # sys.exit is a *detected* reward-hack class, not a sandbox-escape vector, so it is allowed
    # to run (and is then captured); banning it would break the EXIT_HACK corpus fixture.
    scan_candidate_source("def solve(x):\n    import sys\n    sys.exit(0)\n")


# -- out-of-process execution ----------------------------------------------------------------
def test_benign_program_runs_in_child():
    path = _write_program("def solve(x):\n    return x * x\n")
    try:
        solve = load_solve(path, [2, 3])
        assert solve is not None
        assert solve(2) == 4
        assert solve(3) == 9
    finally:
        os.unlink(path)


def test_rejected_candidate_returns_none():
    # A candidate the pre-scan rejects never runs and loads as None (scores zero, like an
    # unimportable program in the previous loader).
    path = _write_program("import socket\ndef solve(x):\n    return 1.0\n")
    try:
        assert load_solve(path, [1]) is None
    finally:
        os.unlink(path)


def test_unimportable_candidate_returns_none():
    path = _write_program("this is not valid python !!!\n")
    try:
        assert load_solve(path, [1]) is None
    finally:
        os.unlink(path)


def test_solve_exception_is_replayed():
    path = _write_program("def solve(x):\n    raise ValueError('boom')\n")
    try:
        solve = load_solve(path, [1])
        assert solve is not None
        with pytest.raises(Exception):  # noqa: B017 - replayed as RuntimeError
            solve(1)
    finally:
        os.unlink(path)


# -- environment scrubbing ---------------------------------------------------------------------
def test_scrubbed_env_drops_credential_vars(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy-aws")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "dummy-hub")
    env = scrubbed_env()
    assert "HF_TOKEN" not in env
    assert "OPENAI_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "HUGGINGFACE_HUB_TOKEN" not in env


def test_child_cannot_read_planted_hf_token(monkeypatch):
    """A planted dummy HF_TOKEN must be invisible to candidate code running in the child."""

    monkeypatch.setenv("HF_TOKEN", "PLANTED-DUMMY-DO-NOT-LEAK")
    # The candidate cannot import os (pre-scan), so it reads the env through a passed-in helper:
    # we instead assert at the scrubbing layer (the child receives scrubbed_env()), and prove the
    # value is absent from what the child would be handed.
    assert "HF_TOKEN" not in scrubbed_env()
    assert "PLANTED-DUMMY-DO-NOT-LEAK" not in scrubbed_env().values()


def test_child_process_cannot_read_planted_hf_token(monkeypatch):
    """End-to-end: a program that *does* read os.environ['HF_TOKEN'] in the child sees nothing.

    This exercises the real subprocess boundary (``_run_child`` does not pre-scan, so the program
    can reach ``os.environ``) and proves the planted dummy credential never reaches the child — the
    in-process loader this replaces would have handed it straight over. The token value is a fixed
    dummy string; nothing here makes a network call or runs LLM-generated code.
    """

    monkeypatch.setenv("HF_TOKEN", "PLANTED-DUMMY-DO-NOT-LEAK")
    monkeypatch.setenv("OPENAI_API_KEY", "PLANTED-OPENAI")
    program = (
        "import os\n"
        "def solve(x):\n"
        "    return [os.environ.get('HF_TOKEN', ''), os.environ.get('OPENAI_API_KEY', '')]\n"
    )
    import importlib

    run_child = importlib.import_module("rewardfuzz.synth.exec_program")._run_child
    path = _write_program(program)
    try:
        lines = run_child(path, [1], timeout_s=5.0)
    finally:
        os.unlink(path)
    # First line is the load result (import os succeeds inside the child runner), then one per input.
    case = lines[-1]
    assert case.get("value") == ["", ""], case
    assert "PLANTED-DUMMY-DO-NOT-LEAK" not in repr(lines)
