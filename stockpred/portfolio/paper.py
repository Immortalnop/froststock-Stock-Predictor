"""Paper trading. `PaperAccount` is a local simulated broker persisted in the store: orders are decided
at a close and filled at the next bar's open with the backtest cost model, so the live forward test
uses the same execution assumptions as the backtest. `AlpacaPaper` submits the same targets to
Alpaca's paper endpoint (needs keys; not exercised by the tests)."""

from __future__ import annotations

import uuid

import pandas as pd

from stockpred.backtest.costs import CostModel
from stockpred.backtest.engine import market_arrays
from stockpred.data.store import Store


class PaperAccount:
    def __init__(self, store: Store, name: str = "default", initial_cash: float = 1_000_000.0,
                 cost: CostModel | None = None):
        self.store, self.name, self.cost = store, name, cost or CostModel()
        with store._lock:
            have = store.con.execute("SELECT 1 FROM paper_cash WHERE account=?", [name]).fetchall()
            if not have:
                store.con.execute("INSERT INTO paper_cash VALUES (?,?,?)", [name, initial_cash, initial_cash])

    # ------------------------------------------------------------ state
    @property
    def cash(self) -> float:
        with self.store._lock:
            return float(self.store.con.execute("SELECT cash FROM paper_cash WHERE account=?",
                                                [self.name]).fetchall()[0][0])

    @property
    def initial(self) -> float:
        with self.store._lock:
            return float(self.store.con.execute("SELECT initial FROM paper_cash WHERE account=?",
                                                [self.name]).fetchall()[0][0])

    def positions(self) -> pd.Series:
        with self.store._lock:
            df = self.store.con.execute("SELECT symbol, qty FROM paper_positions WHERE account=? AND qty<>0",
                                        [self.name]).df()
        return pd.Series(df["qty"].to_numpy(), index=df["symbol"], dtype=float)

    def nav_history(self) -> pd.DataFrame:
        with self.store._lock:
            df = self.store.con.execute("SELECT date, nav, cash, gross, net FROM paper_nav WHERE account=? "
                                        "ORDER BY date", [self.name]).df()
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def orders(self, status: str | None = None) -> pd.DataFrame:
        q, a = "SELECT * FROM paper_orders WHERE account=?", [self.name]
        if status:
            q, a = q + " AND status=?", a + [status]
        with self.store._lock:
            return self.store.con.execute(q + " ORDER BY decided_on, symbol", a).df()

    # ---------------------------------------------------------- trading
    def _closes(self, asof: pd.Timestamp) -> pd.Series:
        px = self.store.prices_df(end=asof)
        return px[px.date == px.date.max()].set_index("symbol")["close"] if len(px) else pd.Series(dtype=float)

    def nav(self, asof: pd.Timestamp) -> float:
        pos, cl = self.positions(), self._closes(asof)
        return self.cash + float((pos * cl.reindex(pos.index).fillna(0)).sum())

    def rebalance(self, target_w: pd.Series, asof: pd.Timestamp) -> pd.DataFrame:
        """Queue orders to reach `target_w` (fractions of NAV) from the close of `asof`; they fill next open.

        Pending orders from an earlier decision are cancelled first, so only the latest intent is live.
        """
        asof = pd.Timestamp(asof).normalize()
        cl = self._closes(asof)
        nav = self.nav(asof)
        pos = self.positions()
        syms = sorted(set(target_w.index) | set(pos.index))
        with self.store._lock:
            self.store.con.execute("UPDATE paper_orders SET status='cancelled' WHERE account=? AND status='pending'",
                                   [self.name])
            rows = []
            for s in syms:
                px = cl.get(s)
                if px is None or not px > 0:
                    continue
                want = float(target_w.get(s, 0.0)) * nav / px
                dq = want - float(pos.get(s, 0.0))
                if abs(dq * px) < 0.0005 * nav:  # ignore dust trades (<5 bp of NAV)
                    continue
                rows.append((uuid.uuid4().hex[:12], self.name, asof.date(), s, float(target_w.get(s, 0.0)), dq,
                             "pending", None, None, None))
            if rows:
                self.store.con.executemany("INSERT INTO paper_orders VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        return self.orders("pending")

    def settle(self) -> int:
        """Fill pending orders at the first open after their decision date. Returns number filled."""
        pend = self.orders("pending")
        if pend.empty:
            return 0
        px = self.store.prices_df(list(pend.symbol.unique()))
        if px.empty:
            return 0
        op, adv, vol = market_arrays(px, list(pend.symbol.unique()))
        raw_open = px.pivot(index="date", columns="symbol", values="open").reindex(op.index)
        n = 0
        with self.store._lock:
            for r in pend.itertuples():
                dec = pd.Timestamp(r.decided_on)
                later = raw_open.index[raw_open.index > dec]
                if not len(later) or pd.isna(raw_open.loc[later[0], r.symbol]):
                    continue
                d = later[0]
                o = float(raw_open.loc[d, r.symbol])
                a = float(adv.loc[d, r.symbol]) if pd.notna(adv.loc[d, r.symbol]) else float("nan")
                v = float(vol.loc[d, r.symbol]) if pd.notna(vol.loc[d, r.symbol]) else 0.0
                notional = abs(r.qty) * o
                frac = float(self.cost.cost_frac(notional, a, v))
                fill = o * (1 + frac) if r.qty > 0 else o * (1 - frac)
                cost = notional * frac
                self.store.con.execute(
                    "UPDATE paper_orders SET status='filled', fill_date=?, fill_price=?, cost=? WHERE order_id=?",
                    [d.date(), fill, cost, r.order_id])
                self.store.con.execute("UPDATE paper_cash SET cash = cash - ? WHERE account=?",
                                       [r.qty * fill, self.name])
                cur = self.store.con.execute("SELECT qty FROM paper_positions WHERE account=? AND symbol=?",
                                             [self.name, r.symbol]).fetchall()
                newq = (cur[0][0] if cur else 0.0) + r.qty
                self.store.con.execute("INSERT OR REPLACE INTO paper_positions VALUES (?,?,?)",
                                       [self.name, r.symbol, newq])
                n += 1
        return n

    def mark(self, asof: pd.Timestamp) -> dict:
        """Record end-of-day NAV, gross and net exposure at the close of `asof`."""
        asof = pd.Timestamp(asof).normalize()
        pos, cl = self.positions(), self._closes(asof)
        mv = pos * cl.reindex(pos.index).fillna(0)
        nav = self.cash + float(mv.sum())
        row = {"nav": nav, "cash": self.cash, "gross": float(mv.abs().sum() / nav), "net": float(mv.sum() / nav)}
        with self.store._lock:
            self.store.con.execute("INSERT OR REPLACE INTO paper_nav VALUES (?,?,?,?,?,?)",
                                   [self.name, asof.date(), nav, row["cash"], row["gross"], row["net"]])
        return row


def alpaca_orders_payload(target_w: pd.Series, nav: float, prices: pd.Series, current_qty: pd.Series) -> list[dict]:
    """Market-on-open orders that move current holdings to target. Pure function (testable offline)."""
    out = []
    for s in sorted(set(target_w.index) | set(current_qty.index)):
        px = prices.get(s)
        if px is None or not px > 0:
            continue
        dq = int(round(float(target_w.get(s, 0.0)) * nav / px - float(current_qty.get(s, 0.0))))
        if dq:
            out.append({"symbol": s, "qty": str(abs(dq)), "side": "buy" if dq > 0 else "sell", "type": "market",
                        "time_in_force": "opg"})
    return out


class AlpacaPaper:
    """Thin client for Alpaca's paper-trading REST API. Paper endpoint only; the live URL is never used."""

    BASE = "https://paper-api.alpaca.markets/v2"

    def __init__(self, key_id: str, secret: str):
        self.h = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}

    async def _req(self, method: str, path: str, **kw):
        import aiohttp

        async with aiohttp.ClientSession(headers=self.h) as s:
            async with s.request(method, self.BASE + path, **kw) as r:
                r.raise_for_status()
                return await r.json()

    async def account(self) -> dict:
        return await self._req("GET", "/account")

    async def positions(self) -> pd.Series:
        p = await self._req("GET", "/positions")
        return pd.Series({x["symbol"]: float(x["qty"]) for x in p}, dtype=float)

    async def submit(self, orders: list[dict]) -> list[dict]:
        await self._req("DELETE", "/orders")  # cancel stale intent first
        return [await self._req("POST", "/orders", json=o) for o in orders]
