"""Security universe: tickers, sectors, aliases used for entity linking.

`default_universe()` is a static large-cap set so everything works offline.
`sp500_membership()` (stockpred.data.universe_sp500) reconstructs point-in-time index
membership to limit survivorship bias in research runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Security:
    symbol: str
    name: str
    sector: str
    aliases: tuple[str, ...] = ()

    @property
    def yahoo(self) -> str:
        return self.symbol.replace(".", "-")


@dataclass
class Universe:
    securities: list[Security] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_symbol = {s.symbol: s for s in self.securities}

    @property
    def symbols(self) -> list[str]:
        return [s.symbol for s in self.securities]

    def get(self, symbol: str) -> Security | None:
        return self._by_symbol.get(symbol)

    def sector_of(self, symbol: str) -> str:
        s = self._by_symbol.get(symbol)
        return s.sector if s else "Unknown"

    def __len__(self) -> int:
        return len(self.securities)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._by_symbol


def _s(sym: str, name: str, sector: str, *aliases: str) -> Security:
    return Security(sym, name, sector, tuple(aliases))


_DEFAULT = [
    _s("AAPL", "Apple", "Technology", "iPhone", "iPad", "Tim Cook", "Mac"),
    _s("MSFT", "Microsoft", "Technology", "Azure", "Windows", "Satya Nadella", "Copilot"),
    _s("GOOGL", "Alphabet", "Communication", "Google", "YouTube", "Waymo", "Gemini"),
    _s("AMZN", "Amazon", "Consumer Discretionary", "AWS", "Amazon Web Services", "Prime"),
    _s("NVDA", "NVIDIA", "Technology", "Jensen Huang", "CUDA", "Blackwell"),
    _s("META", "Meta Platforms", "Communication", "Facebook", "Instagram", "WhatsApp", "Zuckerberg"),
    _s("TSLA", "Tesla", "Consumer Discretionary", "Elon Musk", "Cybertruck", "Model Y"),
    _s("BRK-B", "Berkshire Hathaway", "Financials", "Berkshire", "Warren Buffett"),
    _s("JPM", "JPMorgan Chase", "Financials", "JPMorgan", "Jamie Dimon"),
    _s("V", "Visa", "Financials"),
    _s("UNH", "UnitedHealth", "Health Care", "UnitedHealth Group"),
    _s("XOM", "Exxon Mobil", "Energy", "ExxonMobil", "Exxon"),
    _s("JNJ", "Johnson & Johnson", "Health Care", "J&J"),
    _s("WMT", "Walmart", "Consumer Staples"),
    _s("PG", "Procter & Gamble", "Consumer Staples", "P&G"),
    _s("HD", "Home Depot", "Consumer Discretionary"),
    _s("BAC", "Bank of America", "Financials", "BofA"),
    _s("KO", "Coca-Cola", "Consumer Staples", "Coca Cola"),
    _s("PFE", "Pfizer", "Health Care"),
    _s("CVX", "Chevron", "Energy"),
    _s("NFLX", "Netflix", "Communication"),
    _s("AMD", "Advanced Micro Devices", "Technology", "AMD"),
    _s("INTC", "Intel", "Technology"),
    _s("CRM", "Salesforce", "Technology"),
    _s("ORCL", "Oracle", "Technology"),
    _s("DIS", "Walt Disney", "Communication", "Disney"),
    _s("BA", "Boeing", "Industrials"),
    _s("CAT", "Caterpillar", "Industrials"),
    _s("GS", "Goldman Sachs", "Financials", "Goldman"),
    _s("COST", "Costco", "Consumer Staples"),
]


def default_universe() -> Universe:
    return Universe(list(_DEFAULT))
