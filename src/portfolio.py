"""Portafoglio simulato: stesse regole per bot e (in futuro) giocatore umano.

Sizing del bot: rischio fisso -> quote = (equity * risk_per_trade) / (entry - stop),
con cap al max_position_pct dell'equity e vincolo di cassa (no leva).
Costi: commissioni + slippage in basis point su ogni lato.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float          # prezzo effettivo (slippage incluso)
    shares: float
    stop: float
    signal_entry: float
    side: str = "long"          # "long" o "short" (short: si guadagna se scende)
    raised: bool = False        # True se lo stop è stato mosso dal trailing
    last_price: float = 0.0     # ultimo prezzo visto: fallback onesto per equity()

    def __post_init__(self) -> None:
        if not self.last_price:
            self.last_price = self.entry_price

    @property
    def sign(self) -> float:
        return 1.0 if self.side == "long" else -1.0


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    reason: str
    pnl: float
    ret: float
    side: str = "long"


@dataclass
class Portfolio:
    initial_capital: float
    risk_per_trade: float
    max_position_pct: float
    max_positions: int
    commission_bps: float
    slippage_bps: float
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_capital

    def _cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10_000.0

    def mark_prices(self, prices: dict[str, float]) -> None:
        """Registra l'ultimo prezzo visto per ogni posizione. Va chiamato a
        ogni settimana PRIMA di equity(), così il fallback resta onesto."""
        for tk, pos in self.positions.items():
            if tk in prices:
                pos.last_price = prices[tk]

    def equity(self, prices: dict[str, float]) -> float:
        """Valore totale: cassa + posizioni ai prezzi correnti.

        Se un prezzo manca (buco nei dati di quella settimana), si usa
        `last_price` — l'ultimo prezzo effettivamente visto — e NON il prezzo
        d'ingresso: valutare al prezzo d'ingresso significherebbe fingere che
        la posizione non abbia mai perso nulla, mascherando i drawdown.
        """
        # long: +shares×prezzo; short: −shares×prezzo (la cassa già include i
        # proventi della vendita allo scoperto, e si deve ricomprare a prezzo)
        held = sum(p.sign * p.shares * prices.get(p.ticker, p.last_price)
                   for p in self.positions.values())
        return self.cash + held

    def can_open(self, ticker: str) -> bool:
        return ticker not in self.positions and len(self.positions) < self.max_positions

    def size_fixed_risk(self, equity: float, entry: float, stop: float,
                        side: str = "long", cash_cap: bool = True) -> float:
        risk_amount = equity * self.risk_per_trade
        # rischio per azione: long = entry−stop (stop sotto); short = stop−entry (stop sopra)
        per_share = (entry - stop) if side == "long" else (stop - entry)
        if per_share <= 0:
            return 0.0
        shares = risk_amount / per_share
        max_notional = equity * self.max_position_pct
        shares = min(shares, max_notional / entry)
        if side == "long" and cash_cap:
            # no leva sul long: vincolo di cassa. Lo short INCASSA i proventi,
            # quindi non consuma cassa; è limitato dal rischio e da max_position.
            # cash_cap=False serve al ribilanciamento (si fa spazio trimmando).
            shares = min(shares, self.cash / (entry * (1 + self._cost_rate())))
        return max(shares, 0.0)

    def reduce(self, ticker: str, date: pd.Timestamp, price: float,
               sell_shares: float, reason: str):
        """Chiude PARZIALMENTE una posizione long: vende `sell_shares`, realizza
        la quota venduta e lascia aperto il resto. Serve al ribilanciamento: fare
        spazio a un nuovo segnale senza chiudere del tutto le posizioni esistenti."""
        pos = self.positions.get(ticker)
        if pos is None or pos.side != "long":
            return None
        sell_shares = min(sell_shares, pos.shares)
        if sell_shares <= 0:
            return None
        fill = price * (1 - self._cost_rate())
        self.cash += sell_shares * fill
        pnl = sell_shares * (fill - pos.entry_price)
        trade = Trade(ticker, pos.entry_date, date, pos.entry_price, fill,
                      sell_shares, reason, pnl, fill / pos.entry_price - 1.0, side="long")
        self.trades.append(trade)
        pos.shares -= sell_shares
        if pos.shares <= 1e-9:
            self.positions.pop(ticker)
        return trade

    def open(self, ticker: str, date: pd.Timestamp, entry: float, stop: float,
             shares: float):
        if shares <= 0 or not self.can_open(ticker):
            return None
        fill = entry * (1 + self._cost_rate())
        cost = shares * fill
        if cost > self.cash + 1e-9:
            return None
        self.cash -= cost
        pos = Position(ticker, date, fill, shares, stop, entry, side="long")
        self.positions[ticker] = pos
        return pos

    def open_short(self, ticker: str, date: pd.Timestamp, entry: float, stop: float,
                   shares: float):
        """Vendita allo scoperto: incassi i proventi ora, ricomprerai dopo.
        Si guadagna se il prezzo scende. Lo stop sta SOPRA l'ingresso."""
        if shares <= 0 or not self.can_open(ticker):
            return None
        fill = entry * (1 - self._cost_rate())      # vendi: ricevi un filo meno
        self.cash += shares * fill                  # proventi della vendita
        pos = Position(ticker, date, fill, shares, stop, entry, side="short")
        self.positions[ticker] = pos
        return pos

    def charge_borrow(self, prices: dict[str, float], weekly_rate: float) -> None:
        """Costo di prestito settimanale sui corti (carry): shares×prezzo×tasso."""
        if weekly_rate <= 0:
            return
        for p in self.positions.values():
            if p.side == "short":
                px = prices.get(p.ticker, p.last_price)
                self.cash -= p.shares * px * weekly_rate

    def close(self, ticker: str, date: pd.Timestamp, price: float, reason: str) -> Trade:
        pos = self.positions.pop(ticker)
        if pos.side == "long":
            fill = price * (1 - self._cost_rate())          # vendi
            self.cash += pos.shares * fill
            pnl = pos.shares * (fill - pos.entry_price)
            ret = fill / pos.entry_price - 1.0
        else:                                                # short: ricopri
            fill = price * (1 + self._cost_rate())          # ricompra
            self.cash -= pos.shares * fill
            pnl = pos.shares * (pos.entry_price - fill)      # guadagno se fill<entry
            ret = pos.entry_price / fill - 1.0
        trade = Trade(ticker, pos.entry_date, date, pos.entry_price, fill,
                      pos.shares, reason, pnl, ret, side=pos.side)
        self.trades.append(trade)
        return trade
