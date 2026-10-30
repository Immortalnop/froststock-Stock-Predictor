# config/settings.py
"""
Central configuration for Stock News Predictor.
Edit this file to customise behaviour without touching core logic.
"""

# ─────────────────────────────────────────────
# LLM SETTINGS
# ─────────────────────────────────────────────
LLM_PROVIDER    = "ollama"               # "ollama" | "openai"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.2"             # any model you have pulled
OLLAMA_TIMEOUT  = 60                     # seconds per request
LLM_TEMPERATURE = 0.1                    # low = deterministic

# ─────────────────────────────────────────────
# NEWS SOURCES  (RSS – no API key needed)
# ─────────────────────────────────────────────
RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "https://www.marketwatch.com/rss/topstories",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.feedburner.com/TechCrunch",
    "https://www.theverge.com/rss/index.xml",
]

# ─────────────────────────────────────────────
# STOCKS TO MONITOR
# ─────────────────────────────────────────────
WATCHLIST = [
    {"symbol": "AAPL",  "name": "Apple",         "sector": "Technology"},
    {"symbol": "MSFT",  "name": "Microsoft",      "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet",       "sector": "Technology"},
    {"symbol": "NVDA",  "name": "NVIDIA",         "sector": "Semiconductors"},
    {"symbol": "META",  "name": "Meta Platforms", "sector": "Technology"},
    {"symbol": "AMZN",  "name": "Amazon",         "sector": "E-Commerce/Cloud"},
    {"symbol": "TSLA",  "name": "Tesla",          "sector": "EV/Energy"},
    {"symbol": "JPM",   "name": "JPMorgan Chase", "sector": "Finance"},
    {"symbol": "BRK-B", "name": "Berkshire",      "sector": "Conglomerate"},
]

# ─────────────────────────────────────────────
# PIPELINE SETTINGS
# ─────────────────────────────────────────────
NEWS_FETCH_INTERVAL_MINUTES  = 5    # how often to poll feeds
MAX_ARTICLES_PER_CYCLE       = 20   # articles analysed per run
MAX_ARTICLE_AGE_HOURS        = 12   # skip articles older than this
SENTIMENT_CACHE_TTL_SECONDS  = 300  # reuse LLM result for 5 min

# ─────────────────────────────────────────────
# PREDICTION WEIGHTS  (must sum to 1.0)
# ─────────────────────────────────────────────
WEIGHT_SENTIMENT      = 0.50
WEIGHT_VOLUME_BUZZ    = 0.20
WEIGHT_RECENCY        = 0.15
WEIGHT_PRICE_MOMENTUM = 0.15

assert abs(
    WEIGHT_SENTIMENT + WEIGHT_VOLUME_BUZZ +
    WEIGHT_RECENCY   + WEIGHT_PRICE_MOMENTUM - 1.0
) < 1e-6, "Weights must sum to 1.0"

# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────
DASHBOARD_REFRESH_SECONDS = 30
LOG_LEVEL = "INFO"   # DEBUG | INFO | WARNING | ERROR
