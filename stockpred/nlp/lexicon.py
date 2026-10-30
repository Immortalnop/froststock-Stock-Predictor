"""Cheap lexicon sentiment baseline (finance-flavoured, Loughran-McDonald-style word lists).

This is a compact hand-curated subset, not the full LM dictionary; swap in the full file via
`load_lexicon(path)` if you have a license. Used as (a) benchmark for the LLM and
(b) cheap scorer for GDELT backfill where no LLM budget exists.
"""

from __future__ import annotations

import re
from pathlib import Path

POSITIVE = set("""
beat beats beating surge surges surged soar soars soared rally rallies rallied jump jumps jumped gain
gains gained rise rises rose record strong stronger strength growth grow grows grew profit profits
profitable upgrade upgraded upgrades outperform outperforms bullish optimism optimistic improve improves
improved improvement boost boosts boosted approval approved approves win wins won breakthrough exceed
exceeds exceeded exceeding raise raised raises dividend buyback expansion accelerate accelerated momentum
recovery recover recovered robust favorable lucrative innovative leadership successful success
""".split())

NEGATIVE = set("""
miss misses missed plunge plunges plunged tumble tumbles tumbled slump slumps slumped fall falls fell drop
drops dropped decline declines declined loss losses lose lost weak weaker weakness cut cuts slash slashed
downgrade downgraded downgrades underperform bearish pessimism pessimistic concern concerns worry worries
lawsuit sue sued probe investigation fraud recall recalled bankruptcy default layoffs layoff fired fine
fined penalty penalties scandal warning warns warned risk risks risky shortfall delay delays delayed
halt halted suspend suspended crisis crash crashes plummet plummets plummeted disappointing disappoints
disappointed probe breach hack hacked antitrust ban banned tariff tariffs
""".split())

NEGATORS = {"not", "no", "never", "without", "neither", "hardly", "barely", "n't", "fails", "failed"}
_TOK = re.compile(r"[a-z']+")


def load_lexicon(path: str | Path) -> tuple[set[str], set[str]]:
    """File format: one `word,+1` or `word,-1` per line."""
    pos, neg = set(), set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "," in line:
            w, s = line.rsplit(",", 1)
            (pos if s.strip().startswith("+") or s.strip() == "1" else neg).add(w.strip().lower())
    return pos, neg


def score_text(text: str, pos: set[str] | None = None, neg: set[str] | None = None) -> float:
    """Return sentiment in [-1, 1]: (P - N) / (P + N + 2), with 3-token negation window."""
    pos, neg = pos or POSITIVE, neg or NEGATIVE
    toks = _TOK.findall(text.lower())
    p = n = 0
    for i, t in enumerate(toks):
        negated = any(x in NEGATORS for x in toks[max(0, i - 3) : i])
        if t in pos:
            n, p = (n + 1, p) if negated else (n, p + 1)
        elif t in neg:
            p, n = (p + 1, n) if negated else (p, n + 1)
    return (p - n) / (p + n + 2)
