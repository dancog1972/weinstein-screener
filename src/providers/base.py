"""Interfaccia comune dei provider di dati.

Tutti i provider restituiscono lo stesso schema di DataFrame giornaliero:
    index: DatetimeIndex (date di borsa, tz-naive)
    colonne: open, high, low, close, adj_close, volume  (float)
`adj_close` è il prezzo rettificato per split/dividendi: medie mobili e RS
usano quello. Cambiare fonte dati = cambiare una riga in config.yaml.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

DAILY_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class DataProvider(ABC):
    """Contratto minimo di un provider EOD."""

    name: str = "base"

    @abstractmethod
    def get_eod(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Storico giornaliero [start, end] nello schema DAILY_COLUMNS."""

    @staticmethod
    def validate(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        missing = [c for c in DAILY_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{ticker}: colonne mancanti dal provider: {missing}")
        df = df[DAILY_COLUMNS].astype(float).sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # --- SANIFICAZIONE ---------------------------------------------------
        # I dati EOD grezzi contengono errori che a valle diventano "breakout"
        # fantasma con rendimenti a quattro-cifre: prezzi a zero, negativi, o
        # assurdi da split-adjustment corrotto. Il fattore adj_close/close
        # esplode quando close≈0, trascinando resistenze a valori come 1000000.
        # Meglio scartare la riga alla fonte che inseguire il problema nel
        # motore. Ogni condizione qui riflette un caso visto nei dati reali.
        n0 = len(df)
        px = ["open", "high", "low", "close", "adj_close"]
        # 1) prezzi non positivi: un titolo non può valere zero o meno
        df = df[(df[px] > 0).all(axis=1)]
        # 2) incoerenza OHLC: il minimo non può superare il massimo
        df = df[df["low"] <= df["high"]]
        # 3) adj_close/close fuori scala: fattore di aggiustamento plausibile
        #    sta fra 0.001 e 1 (riduce all'indietro). Oltre = dato corrotto.
        ratio = df["adj_close"] / df["close"]
        df = df[(ratio > 0.0005) & (ratio <= 1.5)]
        # 4) salti di prezzo impossibili settimana su settimana (>1000% o
        #    <-95% in un giorno non aggiustato sono quasi sempre errori)
        ret = df["adj_close"].pct_change()
        df = df[ret.isna() | ((ret > -0.95) & (ret < 10))]

        # 5) SPIKE intragiornalieri anomali. Un `low` molto sotto il CORPO della
        #    barra (min di open/close) — o un `high` molto sopra — che rientra
        #    entro la chiusura è quasi sempre un tick corrotto, non un movimento
        #    reale (un crollo vero CHIUDE vicino al minimo, non recupera). Questi
        #    low sono "coerenti" (low<=high) quindi sfuggono al controllo 2, ma a
        #    valle avvelenano minimo settimanale, pivot, base e ATR, e provocano
        #    uscite false sullo stop. Non scartiamo la barra (open/close/adj sono
        #    buoni): CLIPPIAMO l'estremo a un wick massimo plausibile. Soglia
        #    prudente (40%): tocca solo gli spike palesi, non la volatilità vera.
        max_wick = 0.40
        body_lo = df[["open", "close"]].min(axis=1)
        body_hi = df[["open", "close"]].max(axis=1)
        df["low"] = df["low"].clip(lower=body_lo * (1.0 - max_wick))
        df["high"] = df["high"].clip(upper=body_hi * (1.0 + max_wick))

        removed = n0 - len(df)
        if removed > n0 * 0.5 and n0 > 20:
            # se scartiamo più di metà delle barre, il titolo è troppo sporco
            # per fidarsene: meglio escluderlo del tutto
            raise ValueError(
                f"{ticker}: {removed}/{n0} barre corrotte (>50%), titolo scartato")
        return df
