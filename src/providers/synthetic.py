"""Provider sintetico: genera serie giornaliere con cicli Weinstein realistici.

Serve per sviluppare e testare il motore senza chiave API. Ogni titolo
attraversa cicli Fase 4 (declino) -> Fase 1 (base laterale a bassa
volatilità) -> breakout con picco di volume -> Fase 2 (avanzata) ->
Fase 3 (top). Deterministico dato il seed: test riproducibili.
Il benchmark 'SYN-BENCH' è la media dei titoli: quando la maggioranza è in
Fase 2 lo è anche il mercato, rendendo sensato il market filter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DataProvider

N_TICKERS = 24
_PHASES = ("decline", "base", "advance", "top")


def _phase_params(phase: str, rng: np.random.Generator) -> tuple[float, float, float]:
    """(drift giornaliero, vol giornaliera, moltiplicatore volume)."""
    if phase == "decline":
        return -0.0013 * rng.uniform(0.7, 1.5), 0.018, 1.0
    if phase == "base":
        return 0.0000, 0.007, 0.6
    if phase == "advance":
        return 0.0016 * rng.uniform(0.8, 1.4), 0.013, 1.3
    return 0.0001, 0.015, 1.1


class SyntheticProvider(DataProvider):
    name = "synthetic"

    def __init__(self) -> None:
        self._cache: dict[str, pd.DataFrame] = {}

    def universe(self) -> list[str]:
        return [f"SYN-{i:03d}" for i in range(N_TICKERS)]

    def get_eod(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        df = self._full(ticker)
        return df.loc[start:end].copy()

    # ------------------------------------------------------------------
    def _full(self, ticker: str) -> pd.DataFrame:
        if ticker in self._cache:
            return self._cache[ticker]
        if ticker == "SYN-BENCH":
            df = self._benchmark()
        else:
            seed = int(ticker.split("-")[1]) + 7
            df = self._simulate(seed)
        self._cache[ticker] = df
        return df

    def _dates(self) -> pd.DatetimeIndex:
        return pd.bdate_range("1998-01-01", "2025-06-30")  # margine pre-2000 per la MA30w

    def _simulate(self, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = self._dates()
        n = len(dates)
        log_price = np.empty(n)
        volume = np.empty(n)
        log_price[0] = np.log(rng.uniform(15, 80))
        base_vol_level = rng.uniform(2e5, 2e6)

        phase_idx = rng.integers(0, 4)
        remaining = 0
        vol_spike_days = 0
        spike_used = True
        LOOKBACK = 150  # ~30 settimane: la "resistenza" che innesca il breakout
        for t in range(n):
            if remaining == 0:
                phase_idx = (phase_idx + 1) % 4
                phase = _PHASES[phase_idx]
                dur = {"decline": (100, 220), "base": (220, 420),
                       "advance": (180, 380), "top": (80, 180)}[phase]
                remaining = int(rng.integers(*dur))
                drift, vol, vmult = _phase_params(phase, rng)
                if phase == "advance":
                    spike_used = False   # il picco partirà AL breakout di prezzo
            if t > 0:
                shock = rng.normal(drift, vol)
                if _PHASES[phase_idx] == "advance" and vol_spike_days > 0:
                    shock += 0.006
                log_price[t] = log_price[t - 1] + shock
                # breakout reale: primo nuovo massimo di ~30 settimane nell'avanzata
                if (_PHASES[phase_idx] == "advance" and not spike_used
                        and log_price[t] > log_price[max(0, t - LOOKBACK):t].max()):
                    vol_spike_days = 15
                    spike_used = True
            v = base_vol_level * vmult * np.exp(rng.normal(0, 0.35))
            if vol_spike_days > 0:
                v *= rng.uniform(2.2, 3.5)
                vol_spike_days -= 1
            volume[t] = v
            remaining -= 1

        close = np.exp(log_price)
        intraday = np.abs(rng.normal(0, 0.008, n))
        high = close * (1 + intraday)
        low = close * (1 - intraday)
        open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.003, n))
        df = pd.DataFrame(
            {"open": open_, "high": np.maximum.reduce([open_, close, high]),
             "low": np.minimum.reduce([open_, close, low]),
             "close": close, "adj_close": close, "volume": volume},
            index=dates,
        )
        return self.validate(df, f"seed={seed}")

    def _benchmark(self) -> pd.DataFrame:
        # media GEOMETRICA delle serie normalizzate: un indice equipesato
        # ribilanciato, che nessun singolo titolo esplosivo può dominare
        parts = [np.log(self._full(t)["adj_close"] / self._full(t)["adj_close"].iloc[0])
                 for t in self.universe()]
        px = np.exp(pd.concat(parts, axis=1).mean(axis=1)) * 1000.0
        vol = pd.Series(1e7, index=px.index)
        df = pd.DataFrame(
            {"open": px, "high": px * 1.004, "low": px * 0.996,
             "close": px, "adj_close": px, "volume": vol}, index=px.index)
        return self.validate(df, "SYN-BENCH")
