"""Tests for report rendering and the CLI."""

from __future__ import annotations

import json

import numpy as np

from rewardfuzz import audit
from rewardfuzz.cli import main
from rewardfuzz.report.render import to_markdown
from rewardfuzz.types import AuditTarget, CandidateKind

GOAL = np.array([1.0, 2.0, 3.0, 4.0])


def _report():
    at = AuditTarget(
        reward_fn=lambda x: float(np.dot(np.asarray(x, dtype=float), GOAL)),
        kind=CandidateKind.VALUE,
        baseline=GOAL.copy(),
        reward_max=30.0,
        name="dot",
    )
    return audit(at, adapter="callable", seed=0)


def test_markdown_render():
    md = to_markdown(_report())
    assert "Hackability" in md
    assert "rewardfuzz audit" in md


def test_cli_list_strategies(capsys):
    rc = main(["list-strategies"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "degenerate" in out
    assert "test_tamper" in out


def test_cli_list_adapters(capsys):
    rc = main(["list-adapters"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "openevolve" in out


def test_cli_bench_quick(capsys):
    rc = main(["bench", "--quick"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "precision=1.000" in out
    assert "discovered: 5/5" in out


def test_cli_bench_writes_json(tmp_path):
    out = tmp_path / "bench.json"
    rc = main(["bench", "--quick", "-o", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["detection"]["overall"]["f1"] == 1.0


def test_cli_audit_module_function(tmp_path, capsys):
    out = tmp_path / "report.json"
    rc = main(
        [
            "audit",
            "rewardfuzz.corpus.targets.numeric:reward_unbounded",
            "--adapter",
            "callable",
            "--reward-max",
            "30",
            "--baseline-json",
            "[1, 2, 3, 4]",
            "-o",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["target"] == "reward_unbounded"
