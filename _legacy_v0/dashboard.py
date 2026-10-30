# ui/dashboard.py
"""
Rich terminal dashboard — live, colour-coded prediction table + news feed.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.predictor import StockSignal

console = Console()


def _score_bar(score: float, half: int = 8) -> Text:
    filled = min(int(abs(score) * half), half)
    t = Text()
    if score >= 0:
        t.append("─" * half,   style="dim")
        t.append("█" * filled, style="bold green")
    else:
        t.append("█" * filled, style="bold red")
        t.append("─" * (half - filled + half), style="dim")
    return t


def _val_style(v: float, hi: float = 1.5) -> str:
    if v >  hi:  return "bold green"
    if v >   0:  return "green"
    if v < -hi:  return "bold red"
    if v <   0:  return "red"
    return "dim white"


_LABEL_STYLE = {
    "STRONG BUY":  "bold bright_green",
    "BUY":         "green",
    "NEUTRAL":     "dim white",
    "SELL":        "red",
    "STRONG SELL": "bold bright_red",
}


def build_signals_table(signals: Dict[str, StockSignal]) -> Table:
    tbl = Table(
        title="📊  Stock Predictions  (LLM Sentiment + Price Momentum)",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="bright_black",
        expand=True,
        show_lines=False,
    )
    tbl.add_column("Ticker",    style="bold white", width=7)
    tbl.add_column("Company",   width=16)
    tbl.add_column("Price",     justify="right", width=8)
    tbl.add_column("Today %",   justify="right", width=8)
    tbl.add_column("Sentiment", justify="right", width=9)
    tbl.add_column("Bar",       width=17)
    tbl.add_column("Score",     justify="right", width=7)
    tbl.add_column("Signal",    width=14)
    tbl.add_column("News",      justify="right", width=5)
    tbl.add_column("Top Insight", no_wrap=False, width=44)

    for sig in sorted(signals.values(), key=lambda s: s.composite_score, reverse=True):
        price_s = f"${sig.current_price:.2f}" if sig.current_price else "N/A"
        today_s = f"{sig.price_change_pct:+.2f}%" if sig.current_price else "—"
        sent_s  = f"{sig.raw_sentiment:+.3f}"
        comp_s  = f"{sig.composite_score:+.3f}"

        tbl.add_row(
            sig.symbol,
            sig.name,
            Text(price_s, style="yellow"),
            Text(today_s, style=_val_style(sig.price_change_pct)),
            Text(sent_s,  style=_val_style(sig.raw_sentiment)),
            _score_bar(sig.composite_score),
            Text(comp_s,  style=_val_style(sig.composite_score)),
            Text(
                f"{sig.prediction_emoji} {sig.prediction_label}",
                style=_LABEL_STYLE.get(sig.prediction_label, "white"),
            ),
            str(sig.article_count),
            Text(sig.top_rationale[:55] or "—", style="dim"),
        )
    return tbl


def build_news_panel(articles: list, max_items: int = 8) -> Panel:
    lines = Text()
    for art in articles[:max_items]:
        age   = art.age_hours
        age_s = f"{int(age * 60)}m" if age < 1 else f"{age:.1f}h"
        lines.append(f"  [{age_s:>4}] ", style="dim")
        lines.append(f"{art.source[:18]:18s}  ", style="cyan dim")
        lines.append(f"{art.title[:72]}\n")
    return Panel(
        lines,
        title="📰  Latest Articles",
        border_style="bright_black",
        padding=(0, 1),
    )


def build_status_panel(
    cycle: int,
    next_secs: int,
    article_count: int,
    model: str,
    healthy: bool,
) -> Panel:
    ts   = datetime.now().strftime("%H:%M:%S")
    icon = "🟢" if healthy else "🔴"
    t = Text()
    t.append(f"  {icon} {model}  ", style="bold" if healthy else "bold red")
    t.append(f"│  Cycle #{cycle}  ")
    t.append(f"│  Articles this run: {article_count}  ")
    t.append(f"│  Refresh in: {next_secs}s  ")
    t.append(f"│  {ts}", style="dim")
    return Panel(t, border_style="bright_black", padding=(0, 0))


def build_picks_panel(signals: Dict[str, StockSignal]) -> Panel:
    ranked = sorted(signals.values(), key=lambda s: s.composite_score, reverse=True)
    t = Text()
    t.append("  🔥 TOP PICKS:  ", style="bold green")
    for sig in ranked[:3]:
        t.append(f"{sig.symbol} ({sig.composite_score:+.2f})  ", style="bold bright_green")
    t.append("     ⚠  WATCH:  ", style="bold red")
    for sig in ranked[-2:]:
        if sig.composite_score < -0.10:
            t.append(f"{sig.symbol} ({sig.composite_score:+.2f})  ", style="bold bright_red")
    return Panel(t, border_style="bright_black", padding=(0, 0))


class Dashboard:
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self._live: Optional[Live] = None

    def start(self):
        self._live = Live(
            console=console,
            refresh_per_second=1,
            screen=True,
        )
        self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()

    def render(
        self,
        signals: Dict[str, StockSignal],
        articles: list,
        cycle: int,
        next_refresh: int,
        healthy: bool,
    ):
        if not self._live:
            return

        article_count = sum(s.article_count for s in signals.values())

        layout = Layout()
        layout.split_column(
            Layout(name="status", size=3),
            Layout(name="picks",  size=3),
            Layout(name="table",  ratio=2),
            Layout(name="news",   size=12),
        )
        layout["status"].update(
            build_status_panel(cycle, next_refresh, article_count, self.model_name, healthy)
        )
        layout["picks"].update(
            build_picks_panel(signals) if signals else Panel("Loading…")
        )
        layout["table"].update(
            build_signals_table(signals) if signals else Panel("Waiting for first data…")
        )
        layout["news"].update(
            build_news_panel(articles) if articles else Panel("Fetching news…")
        )
        self._live.update(layout)

    def print_summary(self, signals: Dict[str, StockSignal]):
        """Fallback for --no-ui mode."""
        console.print(build_signals_table(signals))
