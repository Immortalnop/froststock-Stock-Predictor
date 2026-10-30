#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║         STOCK NEWS PREDICTOR — LOCAL LLM EDITION    ║
║   Realtime news → LLM sentiment → Price prediction  ║
╚══════════════════════════════════════════════════════╝

Usage:
    python main.py                    # Start with default settings
    python main.py --model mistral    # Use a different Ollama model
    python main.py --once             # Single scan, then exit
    python main.py --no-ui            # Plain text output (no live dashboard)
    python main.py --list-models      # Show available Ollama models

Requirements:
    pip install -r requirements.txt
    ollama pull llama3.2   # or any model you prefer
"""

import argparse
import asyncio
import logging
import sys
import time

import aiohttp


# ── Local imports ──────────────────────────────────────────────
from config.settings import (
    RSS_FEEDS,
    WATCHLIST,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    LLM_TEMPERATURE,
    SENTIMENT_CACHE_TTL_SECONDS,
    NEWS_FETCH_INTERVAL_MINUTES,
    MAX_ARTICLES_PER_CYCLE,
    MAX_ARTICLE_AGE_HOURS,
    WEIGHT_SENTIMENT,
    WEIGHT_VOLUME_BUZZ,
    WEIGHT_RECENCY,
    WEIGHT_PRICE_MOMENTUM,
    DASHBOARD_REFRESH_SECONDS,
    LOG_LEVEL,
)
from core.news_fetcher import NewsFetcher
from core.llm_analyzer import LLMAnalyzer
from core.predictor import Predictor
from ui.dashboard import Dashboard, console


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("predictor.log"), logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

async def list_ollama_models(base_url: str):
    """Print available Ollama models and exit."""
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(f"{base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                data = await resp.json()
                models = [m["name"] for m in data.get("models", [])]
                console.print("\n[bold cyan]Available Ollama models:[/bold cyan]")
                for m in models:
                    console.print(f"  • {m}")
                if not models:
                    console.print("[yellow]No models found. Run: ollama pull llama3.2[/yellow]")
    except Exception as e:
        console.print(f"[red]Cannot reach Ollama at {base_url}: {e}[/red]")
        console.print("[dim]Make sure Ollama is running: ollama serve[/dim]")


def print_startup_banner(model: str, feeds: int, stocks: int):
    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║    STOCK NEWS PREDICTOR — LLM EDITION   ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════╝[/bold cyan]")
    console.print(f"  Model    : [green]{model}[/green] (local Ollama)")
    console.print(f"  Feeds    : [yellow]{feeds}[/yellow] RSS sources")
    console.print(f"  Watchlist: [yellow]{stocks}[/yellow] stocks")
    console.print(f"  Interval : [yellow]{NEWS_FETCH_INTERVAL_MINUTES}[/yellow] minutes")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline(
    fetcher: NewsFetcher,
    analyzer: LLMAnalyzer,
    predictor: Predictor,
    tickers: list,
    max_articles: int,
) -> tuple:
    """
    One full pipeline cycle:
      1. Fetch news
      2. Analyze with LLM
      3. Build signals
    Returns (signals, articles)
    """
    # Step 1: Fetch
    articles = await fetcher.fetch_all()
    articles = articles[:max_articles]

    if not articles:
        logger.warning("No articles fetched this cycle.")
        return predictor.get_signals(), []

    # Step 2: LLM analysis
    sentiment_results = await analyzer.analyze_batch(articles, tickers)

    # Step 3: Build predictions
    signals = predictor.build_signals(sentiment_results, articles)

    return signals, articles


async def main_loop(args):
    model = args.model or OLLAMA_MODEL

    if args.list_models:
        await list_ollama_models(OLLAMA_BASE_URL)
        return

    print_startup_banner(model, len(RSS_FEEDS), len(WATCHLIST))

    # ── Init components ──────────────────────────────────────
    fetcher = NewsFetcher(feeds=RSS_FEEDS, max_age_hours=MAX_ARTICLE_AGE_HOURS)
    analyzer = LLMAnalyzer(
        provider="ollama",
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        timeout=OLLAMA_TIMEOUT,
        cache_ttl=SENTIMENT_CACHE_TTL_SECONDS,
    )
    predictor = Predictor(
        watchlist=WATCHLIST,
        weight_sentiment=WEIGHT_SENTIMENT,
        weight_buzz=WEIGHT_VOLUME_BUZZ,
        weight_recency=WEIGHT_RECENCY,
        weight_momentum=WEIGHT_PRICE_MOMENTUM,
    )
    tickers = [s["symbol"] for s in WATCHLIST]

    # ── Health check ─────────────────────────────────────────
    console.print("[dim]Checking Ollama connection…[/dim]")
    healthy = await analyzer.check_ollama_health()
    if not healthy:
        console.print(f"\n[bold red]⚠  Ollama not ready![/bold red]")
        console.print(f"   Model  : [cyan]{model}[/cyan]")
        console.print(f"   Server : [cyan]{OLLAMA_BASE_URL}[/cyan]")
        console.print("\n  [dim]Fix steps:[/dim]")
        console.print("  1. Start Ollama:        [green]ollama serve[/green]")
        console.print(f"  2. Pull the model:      [green]ollama pull {model}[/green]")
        console.print("  3. Re-run this script.\n")
        if not args.force:
            await fetcher.close()
            await analyzer.close()
            return

    # ── Dashboard setup ───────────────────────────────────────
    dash = Dashboard(model_name=model) if not args.no_ui else None

    if dash:
        dash.start()

    cycle = 0
    signals = {}
    articles = []
    last_run = 0

    try:
        while True:
            now = time.time()
            interval = NEWS_FETCH_INTERVAL_MINUTES * 60

            if now - last_run >= interval:
                cycle += 1
                last_run = now
                logger.info("Starting pipeline cycle #%d", cycle)

                try:
                    signals, articles = await run_pipeline(
                        fetcher, analyzer, predictor, tickers, MAX_ARTICLES_PER_CYCLE
                    )
                except Exception as exc:
                    logger.error("Pipeline error: %s", exc)

                if args.no_ui:
                    dash_tmp = Dashboard(model_name=model)
                    dash_tmp.print_summary(signals)

                if args.once:
                    break

            # Update dashboard
            next_refresh = max(0, int(interval - (time.time() - last_run)))
            if dash:
                dash.render(signals, articles, cycle, next_refresh, healthy)

            await asyncio.sleep(DASHBOARD_REFRESH_SECONDS)

    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]Shutting down…[/yellow]")
    finally:
        if dash:
            dash.stop()
        await fetcher.close()
        await analyzer.close()

    console.print("[green]Done.[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time stock prediction via LLM news analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",        type=str,  default=None,  help="Ollama model name (default: from settings.py)")
    parser.add_argument("--once",         action="store_true",       help="Run a single cycle and exit")
    parser.add_argument("--no-ui",        action="store_true",       help="Disable live dashboard (plain text output)")
    parser.add_argument("--force",        action="store_true",       help="Continue even if Ollama health check fails")
    parser.add_argument("--list-models",  action="store_true",       help="List available Ollama models and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_loop(args))
