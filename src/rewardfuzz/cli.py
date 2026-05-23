"""Command-line interface: ``rewardfuzz <subcommand>``.

Subcommands:
  audit            audit a reward function given as ``module:function``
  bench            run the ground-truth benchmark (source of the README numbers)
  report           pretty-print a saved JSON report (``--md`` for Markdown)
  harden           print hardening suggestions from a saved JSON report
  list-strategies  list available attack strategies
  list-adapters    list available adapters
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

from . import __version__
from .adapters import list_adapters
from .audit import audit
from .strategies import list_strategies


def _load_callable(spec: str):
    if ":" not in spec:
        raise SystemExit(f"--target must be 'module:function', got {spec!r}")
    mod_name, _, fn_name = spec.partition(":")
    # The module name is supplied by the operator on their own command line to point at their own
    # reward function — this is the intended use of the CLI, not untrusted external input.
    module = importlib.import_module(mod_name)  # nosemgrep: non-literal-import
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise SystemExit(f"{spec!r} did not resolve to a callable")
    return fn


def _cmd_audit(args: argparse.Namespace) -> int:
    from .report.render import render_console, to_markdown

    target = _load_callable(args.target)
    baseline = json.loads(args.baseline_json) if args.baseline_json else None
    report = audit(
        target,
        adapter=args.adapter,
        kind=args.kind,
        baseline=baseline,
        reward_max=args.reward_max,
        higher_is_better=not args.lower_is_better,
        seed=args.seed,
        budget=args.budget,
        judge="structural+llm" if args.llm else "structural",
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(report.to_json())
        print(f"wrote {args.output}")
    if args.md:
        print(to_markdown(report))
    else:
        render_console(report)
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_bench

    result = run_bench(quick=args.quick, seed=args.seed)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"wrote {args.output}")
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_bench(result)
    return 0


def _print_bench(result: dict[str, Any]) -> None:
    corpus = result["corpus"]
    det = result["detection"]["overall"]
    disc = result["discovery"]
    acc = det["accuracy"]
    print("rewardfuzz benchmark")
    print("=" * 60)
    print(
        f"corpus: {corpus['targets']} targets "
        f"({corpus['gameable_targets']} gameable / {corpus['robust_targets']} robust), "
        f"{corpus['candidates']} labelled candidates "
        f"({corpus['hack_candidates']} hack / {corpus['honest_candidates']} honest)"
    )
    print("\ndetection (judge vs ground-truth labels):")
    print(
        f"  precision={det['precision']:.3f}  recall={det['recall']:.3f}  f1={det['f1']:.3f}"
        f"  (tp={det['tp']} fp={det['fp']} fn={det['fn']} tn={det['tn']})"
    )
    print(f"  accuracy={acc['point']:.3f}  95% CI [{acc['lo']:.3f}, {acc['hi']:.3f}]")
    print("  by domain:")
    for dom, m in result["detection"]["by_domain"].items():
        print(
            f"    {dom:8s} precision={m['precision']:.2f} recall={m['recall']:.2f} f1={m['f1']:.2f} (n={m['n']})"
        )
    print("\ndiscovery (full audit per target):")
    print(
        f"  gameable discovered: {disc['discovered']}/{disc['gameable_targets']} "
        f"(rate={disc['discovery_rate']})   false alarms on robust: "
        f"{disc['false_alarms']}/{disc['robust_targets']} (rate={disc['false_alarm_rate']})"
    )
    print(
        f"  mean Hackability: gameable={disc['mean_hackability_gameable']}  "
        f"robust={disc['mean_hackability_robust']}"
    )
    for t in disc["per_target"]:
        mark = "HACKED" if t["found_exploit"] else "clean "
        print(
            f"    [{mark}] {t['target']:28s} hackability={t['hackability']:5.1f} "
            f"{t['risk_label']:12s} exploits={t['n_hacks']}/{t['n_candidates']}"
        )


def _cmd_report(args: argparse.Namespace) -> int:
    from .report.render import render_console, to_markdown
    from .report.schema import AuditReport, Finding

    with open(args.report, encoding="utf-8") as fh:
        data = json.load(fh)
    data["findings"] = [Finding(**f) for f in data.get("findings", [])]
    report = AuditReport(**data)
    if args.md:
        print(to_markdown(report))
    else:
        render_console(report)
    return 0


def _cmd_harden(args: argparse.Namespace) -> int:
    with open(args.report, encoding="utf-8") as fh:
        data = json.load(fh)
    tips = data.get("hardening", [])
    if not tips:
        print("No hardening suggestions (no exploits found).")
        return 0
    print(f"Hardening suggestions for {data.get('target', '?')}:")
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rewardfuzz", description=__doc__)
    parser.add_argument("--version", action="version", version=f"rewardfuzz {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("audit", help="audit a reward function (module:function)")
    pa.add_argument("target", help="reward function as 'module:function'")
    pa.add_argument("--adapter", default="callable", choices=list_adapters())
    pa.add_argument("--kind", default=None, choices=["value", "program", "response"])
    pa.add_argument(
        "--baseline-json", default=None, help="honest baseline payload as a JSON literal"
    )
    pa.add_argument("--reward-max", type=float, default=None)
    pa.add_argument("--lower-is-better", action="store_true")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--budget", type=int, default=8)
    pa.add_argument(
        "--llm", action="store_true", help="enable the optional LLM strategy/judge (needs HF_TOKEN)"
    )
    pa.add_argument("-o", "--output", default=None, help="write JSON report to this path")
    pa.add_argument("--md", action="store_true", help="print Markdown instead of a rich table")
    pa.set_defaults(func=_cmd_audit)

    pb = sub.add_parser("bench", help="run the ground-truth benchmark")
    pb.add_argument("--quick", action="store_true", help="fewer bootstrap iterations (CI)")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("-o", "--output", default=None)
    pb.add_argument("--json", action="store_true", help="print the full result as JSON")
    pb.set_defaults(func=_cmd_bench)

    pr = sub.add_parser("report", help="render a saved JSON report")
    pr.add_argument("report")
    pr.add_argument("--md", action="store_true")
    pr.set_defaults(func=_cmd_report)

    ph = sub.add_parser("harden", help="print hardening suggestions from a saved report")
    ph.add_argument("report")
    ph.set_defaults(func=_cmd_harden)

    pls = sub.add_parser("list-strategies", help="list attack strategies")
    pls.set_defaults(func=lambda a: print("\n".join(list_strategies())) or 0)

    pla = sub.add_parser("list-adapters", help="list adapters")
    pla.set_defaults(func=lambda a: print("\n".join(list_adapters())) or 0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
