"""Provider EODHD (https://eodhd.com) — piano 'EOD Historical Data — All World'.

La chiave API viene letta dalla variabile d'ambiente EODHD_API_KEY: mai nel
codice, mai nel config (igiene necessaria in vista del deploy cloud).
Una chiamata all'endpoint /eod restituisce l'intero storico del ticker,
quindi la costruzione della cache è economica in termini di chiamate.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

from .base import DataProvider

API_ROOT = "https://eodhd.com/api"


class EODHDProvider(DataProvider):
    name = "eodhd"

    def __init__(self, api_key: str | None = None, throttle_s: float = 0.15) -> None:
        self.api_key = api_key or os.environ.get("EODHD_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "Chiave EODHD assente: esporta EODHD_API_KEY prima di usare provider=eodhd."
            )
        self.throttle_s = throttle_s
        self._session = requests.Session()

    def get_eod(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        url = f"{API_ROOT}/eod/{ticker}"
        params = {"api_token": self.api_key, "fmt": "json",
                  "period": "d", "from": start, "to": end}
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            raise ValueError(f"{ticker}: nessun dato EODHD in [{start}, {end}]")
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").rename(columns={"adjusted_close": "adj_close"})
        time.sleep(self.throttle_s)
        return self.validate(df, ticker)

    def bulk_last_day(self, exchange: str, date: str | None = None,
                      kind: str | None = None) -> list[dict]:
        """TUTTI i titoli di una borsa per UN giorno in UNA sola chiamata API.
        È il cuore dell'aggiornamento incrementale: invece di ~7.000 chiamate
        (una per ticker) ne basta 1 per borsa per giorno.

        kind:
          None         -> prezzi EOD (code, o/h/l/c, adjusted_close, volume)
          'splits'     -> split del giorno (code, split)
          'dividends'  -> dividendi del giorno (code, dividend)
        I feed split/dividendi segnalano quali ticker hanno avuto un'azione
        societaria (quindi lo storico aggiustato è cambiato e va RI-scaricato:
        appendere l'ultima barra creerebbe un gradino nella serie).
        `date` in formato YYYY-MM-DD; se assente, l'ultimo giorno disponibile.
        """
        url = f"{API_ROOT}/eod-bulk-last-day/{exchange}"
        params = {"api_token": self.api_key, "fmt": "json"}
        if date:
            params["date"] = date
        if kind:
            params["type"] = kind
        resp = self._session.get(url, params=params, timeout=120)
        resp.raise_for_status()
        time.sleep(self.throttle_s)
        return resp.json() or []

    # ------------------------------------------------------------------
    def exchange_symbols(self, exchange: str = "US",
                         delisted: bool = False) -> pd.DataFrame:
        """Lista dei ticker di una borsa. Disponibile anche nel piano base.

        `delisted=True` restituisce i titoli non più quotati: sono LORO a
        eliminare il survivorship bias, perché il backtest deve poterli
        comprare finché erano vivi.

        Nota su `_old`: quando un'azienda viene delistata perde il ticker, che
        può essere riassegnato a un'altra società. EODHD marca il vecchio con
        suffisso `_old` (es. ACR_old.US). Sono aziende DIVERSE: il flag
        `is_delisted` permette di trattarle come tali.
        """
        url = f"{API_ROOT}/exchange-symbol-list/{exchange}"
        params = {"api_token": self.api_key, "fmt": "json"}
        if delisted:
            params["delisted"] = "1"
        resp = self._session.get(url, params=params, timeout=120)
        resp.raise_for_status()
        rows = resp.json() or []
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.rename(columns=str.lower)

        # ATTENZIONE alla differenza fra due cose che sembrano la stessa:
        #   - il CODICE DI RICHIESTA dell'API: "US" (copre tutte le borse USA)
        #   - la BORSA REALE del titolo: "NYSE", "NASDAQ", "BATS", "OTC"...
        # Il campo `exchange` restituito da EODHD contiene la SECONDA. Costruire
        # il ticker con quella produce "AAPL.NASDAQ", che l'API non conosce:
        # ogni chiamata risponde 404. Il ticker giusto è "AAPL.US".
        df["exchange_real"] = df.get(
            "exchange", pd.Series(exchange, index=df.index)).astype(str)
        df["ticker"] = df["code"].astype(str) + "." + exchange.upper()
        df["is_delisted"] = delisted
        time.sleep(self.throttle_s)
        return df

    def historical_constituents(self, index: str = "GSPC.INDX") -> pd.DataFrame:
        """Composizione STORICA dell'indice: chi c'era e quando.

        Richiede un piano con i Fundamentals. Per l'S&P 500 EODHD copre da
        gennaio 2000, delistati inclusi — è ciò che permette di eliminare il
        survivorship bias invece di dichiararlo e basta.

        Ritorna un DataFrame con: ticker, start (ingresso nell'indice),
        end (uscita, NaT se ancora dentro), is_delisted.
        """
        url = f"{API_ROOT}/fundamentals/{index}"
        params = {"api_token": self.api_key, "fmt": "json",
                  "filter": "HistoricalTickerComponents"}
        resp = self._session.get(url, params=params, timeout=60)
        if resp.status_code == 403:
            raise RuntimeError(
                f"403 su {index}: il tuo piano non include i Fundamentals.\n"
                "  I componenti storici richiedono un piano con Fundamental Data.\n"
                "  Senza, usa universe_mode: static (composizione attuale, "
                "con survivorship bias dichiarato)."
            )
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            raise ValueError(f"{index}: nessun componente storico restituito")

        rows = []
        for item in (raw.values() if isinstance(raw, dict) else raw):
            if not isinstance(item, dict):
                continue
            code, exch = item.get("Code"), item.get("Exchange", "US")
            if not code:
                continue
            rows.append({
                "ticker": f"{code}.{exch}",
                "start": pd.to_datetime(item.get("StartDate"), errors="coerce"),
                "end": pd.to_datetime(item.get("EndDate"), errors="coerce"),
                "is_delisted": bool(item.get("IsDelisted", False)),
                "is_active": bool(item.get("IsActiveNow", False)),
            })
        df = pd.DataFrame(rows).dropna(subset=["ticker"])
        time.sleep(self.throttle_s)
        return df

    def current_constituents(self, index: str) -> pd.DataFrame:
        """Composizione ATTUALE dell'indice. Disponibile per ~100 indici
        mondiali (anche europei), ma FOTOGRAFIA DI OGGI: usarla su 20 anni
        di storia reintroduce il survivorship bias dalla porta di servizio."""
        url = f"{API_ROOT}/fundamentals/{index}"
        params = {"api_token": self.api_key, "fmt": "json", "filter": "Components"}
        resp = self._session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        raw = resp.json() or {}
        rows = [{"ticker": f"{v['Code']}.{v.get('Exchange','US')}",
                 "name": v.get("Name"), "sector": v.get("Sector")}
                for v in (raw.values() if isinstance(raw, dict) else raw)
                if isinstance(v, dict) and v.get("Code")]
        time.sleep(self.throttle_s)
        return pd.DataFrame(rows)
