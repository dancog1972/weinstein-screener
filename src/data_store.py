"""Cache locale dei dati: un file Parquet per ticker, per provider.

Il fetch avviene una sola volta; i run successivi leggono dal disco.

Due livelli di cache:
  - GIORNALIERO grezzo (dal provider, una volta sola)
  - SETTIMANALE ARRICCHITO (to_weekly + enrich_weekly + stages): il calcolo
    più costoso, ~90% del tempo di un backtest. Dipende solo dai dati grezzi
    e dai parametri di STAGE, non dalle soglie di segnale — quindi variare
    volume_ratio_min o gli stop NON lo invalida, e le prove costano minuti.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .providers.base import DataProvider


class DataStore:
    def __init__(self, data_dir: str | Path, provider: DataProvider) -> None:
        self.provider = provider
        self.root = Path(data_dir) / provider.name
        self.root.mkdir(parents=True, exist_ok=True)
        self.weekly_root = Path(data_dir) / provider.name / "_weekly"
        self.weekly_root.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_").replace("\\", "_")
        return self.root / f"{safe}.parquet"

    def _meta_path(self, ticker: str) -> Path:
        return self._path(ticker).with_suffix(".meta.json")

    def _weekly_path(self, ticker: str, key: str) -> Path:
        safe = ticker.replace("/", "_").replace("\\", "_")
        return self.weekly_root / f"{safe}.{key}.parquet"

    # Versione dello SCHEMA del settimanale arricchito. VA INCREMENTATA quando
    # cambia la logica o l'insieme di colonne prodotte da enrich_weekly /
    # prepare_ticker / base_features / stages. È inclusa nella weekly_key: senza,
    # una modifica al calcolo NON invalida la cache e file stale vengono riusati
    # in silenzio. È già successo: una cache scritta prima di `after_decline` e
    # `adj_open` disattivava senza accorgersene `require_decline` e l'esecuzione
    # `next_open` (look-ahead reintrodotto). Bump = la cache si rigenera da sola.
    #   v1 → schema storico (senza after_decline / adj_open)
    #   v2 → after_decline, adj_open
    #   v3 → base_vol_ratio (contrazione del volume nella base)
    #   v4 → clip degli spike intragiornalieri anomali (providers/base.py)
    #   v5 → base_vol_ratio esclude la coda pre-breakout (dry-up iniziale/centrale)
    #   v6 → base_obv (accumulo nella base) per il filtro volume a due livelli
    #   v7 → top_features nel settimanale (support_est, top_vol_ratio, top_obv…) per lo SHORT
    WEEKLY_SCHEMA_VERSION = 7

    @staticmethod
    def weekly_key(bench_ticker: str, ma_weeks: int, slope_lookback: int,
                   flat_slope: float, start: str, end: str) -> str:
        """Chiave che identifica una configurazione di settimanale arricchito.
        Include ciò che cambia il RISULTATO del calcolo settimanale: versione
        dello schema, benchmark, parametri MA/slope, periodo. NON le soglie di
        segnale (variarle riusa la cache)."""
        raw = (f"v{DataStore.WEEKLY_SCHEMA_VERSION}|{bench_ticker}|{ma_weeks}|"
               f"{slope_lookback}|{flat_slope}|{start}|{end}")
        return hashlib.md5(raw.encode()).hexdigest()[:12]  # noqa: S324

    def get_weekly_enriched(self, ticker: str, start: str, end: str, key: str,
                            compute_fn) -> pd.DataFrame:
        """Settimanale arricchito, da cache se disponibile. `compute_fn` prende
        il giornaliero e restituisce il settimanale arricchito; viene chiamata
        solo in caso di cache miss."""
        wp = self._weekly_path(ticker, key)
        if wp.exists():
            try:
                return pd.read_parquet(wp)
            except Exception:  # noqa: BLE001
                wp.unlink(missing_ok=True)   # cache corrotta: ricalcola
        daily = self.get_with_warmup(ticker, start, end)
        wk = compute_fn(daily)
        try:
            wk.to_parquet(wp)
        except Exception:  # noqa: BLE001, S110
            pass          # se non si può scrivere la cache, pazienza
        return wk

    def get_with_warmup(self, ticker: str, start: str, end: str,
                        warmup_weeks: int = 80) -> pd.DataFrame:
        """Serie che INIZIA warmup_weeks prima di start: gli indicatori (MA30,
        RS52) maturano prima del periodo di backtest, senza NaN dentro il periodo.

        Sulla validità della cache. Un titolo può semplicemente NON avere dati
        prima di una certa data: quotato nel 2010, oppure il provider parte dal
        2000 mentre il warmup vorrebbe risalire al 1999. Pretendere che la cache
        arrivi sempre fino a `pad_start` la dichiarerebbe insufficiente per
        sempre, riscaricando l'intero universo a ogni run — ore di chiamate API
        buttate. Registriamo in un file `.meta.json` da quale data abbiamo
        CHIESTO i dati: se abbiamo già chiesto da lì (o da prima), quel che c'è
        è tutto quel che il provider aveva.
        """

        pad_start = (pd.Timestamp(start) - pd.Timedelta(weeks=warmup_weeks)).strftime("%Y-%m-%d")
        path, meta_path = self._path(ticker), self._meta_path(ticker)
        if path.exists():
            df = pd.read_parquet(path)
            # La cache può contenere dati scaricati PRIMA della sanificazione.
            # Applicarla in lettura evita di riscaricare 20.000 titoli: i
            # prezzi corrotti (zero, split-adjustment esploso) vengono ripuliti
            # ora. Se il titolo è troppo sporco, validate solleva e chi chiama
            # lo salta. `getattr` per robustezza verso provider minimali (test).
            _validate = getattr(self.provider, "validate", None)
            if callable(_validate):
                df = _validate(df, ticker)
            asked_from = None
            asked_to = None
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    asked_from = meta.get("asked_from")
                    asked_to = meta.get("asked_to")
                except Exception:  # noqa: BLE001, S110
                    pass
            if len(df):
                # Un DELISTATO ha dati solo fino alla sua morte: un titolo
                # sparito nel 2015 non "manca dei dati 2016-2024", semplicemente
                # non esiste più. Se abbiamo GIÀ CHIESTO fino a `end`, quel che
                # c'è è tutto quel che il provider aveva: la cache è completa.
                # Senza questa riga, ogni run richiama l'API per migliaia di
                # delistati a caccia di dati inesistenti — ore di rete sprecate.
                covers_end = (
                    df.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=10)
                    or (asked_to is not None
                        and pd.Timestamp(asked_to) >= pd.Timestamp(end) - pd.Timedelta(days=10))
                )
                covers_start = (
                    df.index.min() <= pd.Timestamp(pad_start)          # copre davvero
                    or (asked_from is not None
                        and pd.Timestamp(asked_from) <= pd.Timestamp(pad_start))
                )
                if covers_end and covers_start:
                    return df.loc[pad_start:end].copy()
        df = self.provider.get_eod(ticker, pad_start, end)
        df.to_parquet(path)
        meta_path.write_text(json.dumps({"asked_from": pad_start, "asked_to": end}))
        return df.loc[pad_start:end].copy()
