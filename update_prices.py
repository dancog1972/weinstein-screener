#!/usr/bin/env python3
"""Aggiornamento INCREMENTALE dei prezzi via bulk EODHD (approccio A).

Invece di riscaricare ~7.000 ticker (una chiamata ciascuno), usa l'endpoint
`eod-bulk-last-day` che restituisce TUTTI i titoli di una borsa per un giorno in
una sola chiamata. Per ogni borsa e per ogni giorno mancante dall'ultimo update:

  - PREZZI bulk        -> le nuove barre giornaliere, APPESE alla cache.
  - SPLITS + DIVIDENDI bulk -> i ticker con un'azione societaria. Per QUESTI lo
    storico aggiustato di EODHD è cambiato: appendere l'ultima barra creerebbe un
    GRADINO nella serie (la stessa discontinuità di adjustment che avvelena MA30,
    resistenze, breakout). Quindi li RI-scarichiamo interi (1 chiamata each).

Costo tipico/settimana: ~6 borse × ~5 giorni × 3 feed ≈ 90 chiamate bulk +
qualche centinaio di re-fetch (i ticker ex-div/split della settimana). ~10× meno
del full refresh, dati sempre coerenti con l'aggiustamento corrente.

Aggiorna SOLO i ticker ATTIVI (quelli che lo screener scansiona). Invalida la
cache settimanale dei ticker toccati, così lo screener la ricalcola sui dati nuovi.

Uso (chiave vera nell'ambiente):
    $env:EODHD_API_KEY = "..."
    python update_prices.py                 # tutti i mercati
    python update_prices.py --only US
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from src.console import setup as _console_setup

_console_setup()

import pandas as pd
import yaml

from src.config import load_config
from src.data_store import DataStore
from src.providers import make_provider
from src.sectors import ETF_PREDECESSOR, SECTOR_ETFS
from screener import active_universe

DAILY_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def _rec_to_row(r: dict) -> dict:
    return {"open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"],
            "adj_close": r.get("adjusted_close", r["close"]), "volume": r.get("volume", 0)}


def _last_cached_date(store: DataStore, tickers: list[str]) -> pd.Timestamp | None:
    """Ultima data presente in cache (campione sui primi ticker: sono tutti
    aggiornati insieme, basta la più recente)."""
    ds = []
    for tk in tickers[:25]:
        p = store._path(tk)
        if not p.exists():
            continue
        try:
            ds.append(pd.read_parquet(p, columns=["close"]).index.max())
        except Exception:  # noqa: BLE001
            pass
    return max(ds) if ds else None


def _invalidate_weekly(store: DataStore, tk: str) -> None:
    safe = tk.replace("/", "_").replace("\\", "_")
    for f in store.weekly_root.glob(f"{safe}.*.parquet"):
        f.unlink(missing_ok=True)


def _append(store: DataStore, tk: str, new_rows: dict, asked_to: str) -> bool:
    """Appende le nuove barre alla cache giornaliera, passando per la STESSA
    sanificazione del percorso normale (provider.validate). Torna True se ha
    scritto qualcosa di nuovo."""
    p = store._path(tk)
    try:
        df = pd.read_parquet(p)
    except Exception:  # noqa: BLE001
        return False
    add = pd.DataFrame.from_dict(new_rows, orient="index")
    add.index = pd.to_datetime(add.index)
    add = add[DAILY_COLUMNS].astype(float)
    add = add[~add.index.isin(df.index)]        # solo barre davvero nuove
    if add.empty:
        return False
    out = pd.concat([df, add]).sort_index()
    try:
        out = store.provider.validate(out, tk)   # sanifica (zero/OHLC/spike…)
    except Exception:  # noqa: BLE001
        return False                              # titolo troppo sporco: lascio com'era
    out.to_parquet(p)
    mp = store._meta_path(tk)
    meta = {}
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except Exception:  # noqa: BLE001
            meta = {}
    meta["asked_to"] = asked_to
    mp.write_text(json.dumps(meta))
    return True


def _refetch(store: DataStore, tk: str, start: str, end: str) -> bool:
    """Ri-scarica lo storico intero (1 chiamata): per i ticker con azione
    societaria, così l'aggiustamento resta coerente su tutta la serie."""
    try:
        df = store.provider.get_eod(tk, start, end)   # già sanificato
    except Exception as e:  # noqa: BLE001
        print(f"    [!] re-fetch {tk} fallito: {e}")
        return False
    df.to_parquet(store._path(tk))
    store._meta_path(tk).write_text(json.dumps({"asked_from": start, "asked_to": end}))
    return True


def update_market(name: str, mspec: dict, cfg: dict, store: DataStore, args) -> None:
    ex = mspec.get("bulk_exchange") or mspec["exchanges"][0]
    tickers = active_universe(cfg, mspec["universe_file"], mspec["exchanges"])
    if not tickers:
        print(f"[{name}] nessun ticker attivo in cache — salto")
        return
    # includi anche BENCHMARK e (per gli US) ETF SETTORE: sono nel bulk della
    # stessa borsa, così restano freschi come i titoli e il market/sector stage
    # non resta indietro di una settimana. Sono già in cache (li scarica il
    # download iniziale / lo screener), qui li si tiene aggiornati.
    extra = [mspec["benchmark"]]
    if ex == "US":
        extra += list(SECTOR_ETFS.values()) + list(ETF_PREDECESSOR.values())
    seen = set(tickers)
    upd = tickers + [t for t in dict.fromkeys(extra) if t not in seen]
    today = pd.Timestamp(date.today())
    last = _last_cached_date(store, tickers)
    if last is None:
        print(f"[{name}] cache vuota: serve prima un download completo (build_universe prices)")
        return
    gap = (today - last).days
    if gap <= 1:
        print(f"[{name}] già aggiornato a {last.date()} ✓")
        return
    if gap > args.max_gap_days:
        print(f"[{name}] gap di {gap} giorni > {args.max_gap_days}: meglio un full refresh "
              f"(build_universe.py prices --from-universe {mspec['universe_file']})")
        return

    # giorni di calendario da recuperare (i non-borsa tornano vuoti dal bulk)
    days = [(last + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, gap + 1)]
    print(f"[{name}] {ex}: {len(tickers)} attivi · recupero {len(days)} giorni "
          f"({last.date()} → {today.date()})", flush=True)

    prices: dict[str, dict] = {}     # ticker -> {data: riga}
    actions: set[str] = set()        # ticker con split/dividendo (da ri-scaricare)
    for d in days:
        for r in store.provider.bulk_last_day(ex, d):
            tk = f"{r['code']}.{r['exchange_short_name']}"
            prices.setdefault(tk, {})[pd.Timestamp(r["date"])] = _rec_to_row(r)
        for kind in ("splits", "dividends"):
            for r in store.provider.bulk_last_day(ex, d, kind):
                actions.add(f"{r['code']}.{r.get('exchange', ex)}")

    start_ref = (today - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
    end_ref = today.strftime("%Y-%m-%d")
    appended = refetched = 0
    for tk in upd:
        if tk in actions:               # azione societaria → storico ri-scaricato
            if _refetch(store, tk, start_ref, end_ref):
                _invalidate_weekly(store, tk)
                refetched += 1
        elif tk in prices:              # solo nuove barre → append
            if _append(store, tk, prices[tk], end_ref):
                _invalidate_weekly(store, tk)
                appended += 1
    print(f"[{name}] {ex}: {appended} aggiornati (append) · {refetched} ri-scaricati "
          f"(azioni societarie) · fino a {today.date()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggiornamento incrementale prezzi via bulk EODHD")
    ap.add_argument("--config", default="config_us_liquidity.yaml")
    ap.add_argument("--markets", default="config_screener_markets.yaml")
    ap.add_argument("--only", default=None, help="aggiorna solo questo mercato")
    ap.add_argument("--years", type=int, default=6,
                    help="profondità del re-fetch dei ticker con azioni societarie")
    ap.add_argument("--max-gap-days", type=int, default=45,
                    help="oltre questo gap conviene un full refresh, non l'append")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.markets, encoding="utf-8") as fh:
        markets = yaml.safe_load(fh)["markets"]
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    print(f"Update incrementale · mercati: {', '.join(markets)}")
    for name, mspec in markets.items():
        if args.only and name != args.only:
            continue
        update_market(name, mspec, cfg, store, args)
    print("\n✓ Update completato. Ora lo screener girerà sui dati freschi.")


if __name__ == "__main__":
    main()
