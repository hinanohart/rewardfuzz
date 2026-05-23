"""Human-readable rendering of an :class:`AuditReport` (rich console + Markdown)."""

from __future__ import annotations

from .schema import AuditReport

_RISK_STYLE = {
    "HIGH RISK": "bold red",
    "MEDIUM RISK": "bold yellow",
    "LOW RISK": "yellow",
    "MINIMAL RISK": "green",
}


def render_console(report: AuditReport) -> None:
    """Print a coloured summary to the terminal using rich."""

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    style = _RISK_STYLE.get(report.risk_label, "white")
    ci = report.hack_rate
    header = (
        f"[bold]{report.target}[/bold]  (adapter: {report.adapter})\n"
        f"Hackability [{style}]{report.hackability}/100  {report.risk_label}[/]   "
        f"Robustness {report.robustness}/100\n"
        f"{report.n_hacks}/{report.n_candidates} candidates exploited "
        f"(hack-rate {ci['point']:.2f}, 95% CI [{ci['lo']:.2f}, {ci['hi']:.2f}])"
    )
    console.print(Panel(header, title="rewardfuzz audit", border_style=style))

    table = Table(title="Per-strategy attack success", show_lines=False)
    table.add_column("strategy")
    table.add_column("attempts", justify="right")
    table.add_column("hacks", justify="right")
    table.add_column("ASR", justify="right")
    table.add_column("severity", justify="right")
    table.add_column("top invariant")
    for s in report.strategies:
        table.add_row(
            s["name"],
            str(s["attempts"]),
            str(s["hacks"]),
            f"{s['asr']:.2f}",
            f"{s['severity_mean']:.2f}",
            str(s["top_invariant"] or "-"),
        )
    console.print(table)

    if report.findings:
        ftable = Table(title="Top findings", show_lines=False)
        ftable.add_column("severity")
        ftable.add_column("invariant")
        ftable.add_column("strategy")
        ftable.add_column("candidate")
        for f in report.findings[:10]:
            ftable.add_row(f.severity, str(f.invariant or "-"), f.strategy, f.candidate_preview)
        console.print(ftable)

    if report.hardening:
        console.print("\n[bold]Hardening suggestions:[/bold]")
        for i, tip in enumerate(report.hardening, 1):
            console.print(f"  {i}. {tip}")


def to_markdown(report: AuditReport) -> str:
    """Render the report as Markdown (used by ``rewardfuzz report --md``)."""

    ci = report.hack_rate
    lines = [
        f"# rewardfuzz audit — {report.target}",
        "",
        f"- **Hackability:** {report.hackability}/100 ({report.risk_label})",
        f"- **Robustness:** {report.robustness}/100",
        f"- **Exploited:** {report.n_hacks}/{report.n_candidates} candidates "
        f"(hack-rate {ci['point']:.2f}, 95% CI [{ci['lo']:.2f}, {ci['hi']:.2f}])",
        f"- **Adapter:** {report.adapter}",
        "",
        "## Per-strategy attack success",
        "",
        "| strategy | attempts | hacks | ASR | severity | top invariant |",
        "|---|--:|--:|--:|--:|---|",
    ]
    for s in report.strategies:
        lines.append(
            f"| {s['name']} | {s['attempts']} | {s['hacks']} | {s['asr']:.2f} | "
            f"{s['severity_mean']:.2f} | {s['top_invariant'] or '-'} |"
        )
    if report.findings:
        lines += ["", "## Top findings", ""]
        for f in report.findings[:10]:
            lines.append(
                f"- **[{f.severity}]** `{f.invariant or '-'}` ({f.strategy}): {f.rationale}"
            )
    if report.hardening:
        lines += ["", "## Hardening suggestions", ""]
        for i, tip in enumerate(report.hardening, 1):
            lines.append(f"{i}. {tip}")
    return "\n".join(lines) + "\n"
