"""Rich terminal dashboard."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from stockpred.models.labels import LABEL_EMOJI

console = Console()
_LABEL_STYLE = {"STRONG BUY": "bold bright_green", "BUY": "green", "NEUTRAL": "dim white",
                "SELL": "red", "STRONG SELL": "bold bright_red"}


def score_bar(x: float, half: int = 8, scale: float = 1.0) -> Text:
    """Symmetric bar around a centre axis; |x|/scale saturates at 1."""
    t = Text()
    if x != x:
        return Text("·" * (2 * half + 1), style="dim")
    n = min(int(round(abs(x) / scale * half)), half)
    neg, pos = (n if x < 0 else 0), (n if x > 0 else 0)
    t.append("─" * (half - neg), style="dim")
    t.append("█" * neg, style="bold red")
    t.append("│", style="white")
    t.append("█" * pos, style="bold green")
    t.append("─" * (half - pos), style="dim")
    return t


def _sty(v: float) -> str:
    return "dim white" if v != v or v == 0 else ("green" if v > 0 else "red")


def signals_table(sig: pd.DataFrame, rationale: dict[str, tuple[str, float]]) -> Table:
    wide = console.width >= 130
    tbl = Table(title="Signals (ranked)", box=box.ROUNDED, header_style="bold cyan",
                border_style="bright_black", expand=True)
    cols = [("Ticker", {"style": "bold", "no_wrap": True}), ("Price", {"justify": "right", "no_wrap": True}),
            ("1d %", {"justify": "right", "no_wrap": True}), ("Score", {"justify": "right", "no_wrap": True}),
            ("", {"no_wrap": True}), ("P(up)", {"justify": "right", "no_wrap": True}),
            ("Signal", {"no_wrap": True})]
    if wide:
        cols[6:6] = [("Exp. excess [interval]", {"justify": "right", "no_wrap": True})]
        cols += [("Sent", {"justify": "right", "no_wrap": True}), ("Ev5d", {"justify": "right", "no_wrap": True}),
                 ("Why", {"overflow": "ellipsis", "no_wrap": True})]
    for name, kw in cols:
        tbl.add_column(name, **kw)
    scale = float(sig["score"].abs().max() or 1.0)
    for sym, r in sig.sort_values("score", ascending=False).iterrows():
        rat = rationale.get(sym, ("", 0.0))[0]
        er = "—" if pd.isna(r.exp_excess_ret) else f"{r.exp_excess_ret:+.2%} [{r.lo:+.1%},{r.hi:+.1%}]"
        row = [
            sym,
            "—" if pd.isna(r.price) else f"${r.price:,.2f}",
            Text("—" if pd.isna(r.chg_pct) else f"{r.chg_pct:+.2f}", style=_sty(r.chg_pct)),
            f"{r.score:+.3f}", score_bar(r.score, half=5 if not wide else 8, scale=scale),
            "—" if pd.isna(r.p_up) else f"{r.p_up:.0%}",
        ]
        sigtxt = Text(f"{LABEL_EMOJI.get(r.label, '')} {r.label}", style=_LABEL_STYLE.get(r.label, "white"))
        if wide:
            row += [er, sigtxt,
                    Text("—" if pd.isna(r.sent_ew1d) else f"{r.sent_ew1d:+.2f}", style=_sty(r.sent_ew1d)),
                    "—" if pd.isna(r.n_events_5d) else f"{int(r.n_events_5d)}", Text(rat[:60], style="dim")]
        else:
            row += [sigtxt]
        tbl.add_row(*row)
    return tbl


def news_panel(articles: list, n: int = 8) -> Panel:
    t = Text()
    for a in articles[:n]:
        age = a.age_hours()
        t.append(f" [{(f'{int(age * 60)}m' if age < 1 else f'{age:.1f}h'):>5}] ", style="dim")
        t.append(f"{a.source[:18]:18s} ", style="cyan dim")
        t.append(a.title[:90] + "\n")
    return Panel(t if articles else Text("no new articles this cycle", style="dim"),
                 title="Latest new articles", border_style="bright_black")


def status_panel(cycle: int, model: str, llm_ok: bool, llm_detail: str, bundle_info: str, next_s: int) -> Panel:
    t = Text()
    t.append(f" LLM {'OK' if llm_ok else 'DOWN'} ", style="bold green" if llm_ok else "bold red")
    t.append(f"({model}{'' if llm_ok else ': ' + llm_detail[:60]})  ")
    t.append(f"| model: {bundle_info}  | cycle {cycle}  | next in {next_s}s  | {datetime.now():%H:%M:%S}", style="dim")
    return Panel(t, border_style="bright_black")


def disclaimer() -> Text:
    return Text("Research tool. Scores are statistical estimates, not financial advice.", style="dim italic")


class Dashboard:
    """Live display without alt-screen so the final frame stays visible after exit."""

    def __init__(self):
        self._live: Live | None = None

    def start(self):
        self._live = Live(console=console, refresh_per_second=1, screen=False)
        self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()

    def update(self, status: Panel, table: Table, news: Panel):
        g = Group(status, table, news, disclaimer())
        if self._live:
            self._live.update(g)
        else:
            console.print(g)
