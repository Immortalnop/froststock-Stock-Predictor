"""Ticker/alias resolution. Used to (a) pre-link articles and (b) reject LLM tickers
that do not actually appear in the text (hallucination guard)."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from stockpred.config.universe import Universe

_CASHTAG = re.compile(r"\$([A-Z]{1,5}(?:[.-][A-Z])?)\b")
_TOKEN = re.compile(r"[A-Za-z0-9&.'-]+")
# Symbols that collide with common English words/abbreviations; need cashtag or name to link.
_AMBIGUOUS = {"V", "BA", "CAT", "HD", "KO", "PG", "GS", "ALL", "A", "ON", "IT", "NOW", "MS"}


class EntityLinker:
    def __init__(self, universe: Universe, fuzzy_threshold: int = 94):
        self.universe = universe
        self.fuzzy_threshold = fuzzy_threshold
        self._name_pat: dict[str, re.Pattern] = {}
        for sec in universe.securities:
            names = {sec.name, *sec.aliases}
            names = {n for n in names if len(n) >= 3 or n.isupper()}
            if not names:
                continue
            alts = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
            self._name_pat[sec.symbol] = re.compile(rf"(?<![A-Za-z0-9])(?:{alts})(?![A-Za-z0-9])", re.I)
        self._sym_pat = {
            s.symbol: re.compile(rf"(?<![A-Za-z0-9$])\b{re.escape(s.symbol)}\b")
            for s in universe.securities
        }
        self._name_lc = {s.symbol: [n.lower() for n in {s.name, *s.aliases} if len(n) >= 5]
                         for s in universe.securities}

    def link(self, text: str, hint: str | None = None) -> set[str]:
        """Tickers evidenced in text. `hint` (e.g. a per-ticker Yahoo feed) is *not* trusted alone."""
        found: set[str] = set()
        for m in _CASHTAG.finditer(text):
            sym = m.group(1).replace(".", "-")
            if sym in self.universe:
                found.add(sym)
        for sym, pat in self._name_pat.items():
            if pat.search(text):
                found.add(sym)
        for sym, pat in self._sym_pat.items():
            if sym in _AMBIGUOUS or len(sym) < 2:
                continue
            if pat.search(text):  # uppercase symbol standing alone, e.g. "NVDA"
                found.add(sym)
        if len(found) < 1:
            found |= self._fuzzy(text)
        return found

    def _fuzzy(self, text: str) -> set[str]:
        out: set[str] = set()
        toks = _TOKEN.findall(text)
        grams = [" ".join(toks[i : i + n]).lower() for n in (1, 2, 3) for i in range(len(toks) - n + 1)]
        for sym, names in self._name_lc.items():
            for g in grams:
                if len(g) < 5:
                    continue
                if any(fuzz.ratio(g, n) >= self.fuzzy_threshold for n in names):
                    out.add(sym)
                    break
        return out

    def verify(self, claimed: list[str], text: str, hint: str | None = None) -> list[str]:
        """Keep only claimed tickers that the text itself supports."""
        evidence = self.link(text, hint)
        return [t for t in claimed if t in evidence]
