#!/usr/bin/env python3
"""Prepara la CACHE CLOUD per il job GitHub: SOLO i ticker attivi + benchmark +
ETF settore, giornaliero trimmato agli ultimi N anni, in un tar.gz.

Cosa ESCLUDE (e perché la cache scende da ~5 GB a ~300-500 MB):
  - i ~14.000 DELISTATI: servono solo al backtest, non allo screener forward.
  - la cache settimanale `_weekly`: si ricalcola a ogni run (la chiave include la
    data di fine, quindi ogni settimana è comunque un miss → inutile spedirla).
  - lo storico oltre N anni: allo screener bastano ~2 anni; 6 danno margine + il
    grafico a 4 anni. Il `.meta.json` si copia AS-IS (asked_from resta 2000, così
    get_with_warmup NON tenta di ri-scaricare la storia che abbiamo tagliato).

Include gli universi (data/universe_*.parquet): piccoli e necessari sul runner ad
active_universe.

Uso (una volta in locale con la cache piena; poi lo rifà il job ogni settimana):
    python pack_cache.py --out data_cache.tar.gz
    gh release create data-cache data_cache.tar.gz -t "cache dati" -n "cache screener"
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Impacchetta la cache cloud (attivi-only, N anni)")
    ap.add_argument("--config", default="config_us_liquidity.yaml")
    ap.add_argument("--markets", default="config_screener_markets.yaml")
    ap.add_argument("--years", type=int, default=6, help="anni di storico da tenere")
    ap.add_argument("--out", default="data_cache.tar.gz")
    ap.add_argument("--stage", default="_cache_stage")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.markets, encoding="utf-8") as fh:
        markets = yaml.safe_load(fh)["markets"]
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))  # per _path/_meta_path
    root = Path(cfg["data_dir"]) / cfg["provider"]
    cutoff = (pd.Timestamp(date.today()) - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")

    # ticker da includere: attivi di ogni mercato + benchmark + ETF settore
    tickers: set[str] = set()
    for mspec in markets.values():
        tickers.update(active_universe(cfg, mspec["universe_file"], mspec["exchanges"]))
        tickers.add(mspec["benchmark"])
    tickers.update(SECTOR_ETFS.values())
    tickers.update(ETF_PREDECESSOR.values())

    stage = Path(args.stage)
    if stage.exists():
        shutil.rmtree(stage)
    dst_daily = stage / cfg["data_dir"] / cfg["provider"]
    dst_daily.mkdir(parents=True, exist_ok=True)

    kept = 0
    for tk in sorted(tickers):
        p = store._path(tk)
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p).loc[cutoff:]        # trim ultimi N anni
            df.to_parquet(dst_daily / p.name)
        except Exception:  # noqa: BLE001
            continue
        mp = store._meta_path(tk)                        # meta AS-IS (asked_from resta)
        if mp.exists():
            shutil.copy(mp, dst_daily / mp.name)
        kept += 1

    # universi (necessari ad active_universe sul runner)
    for mspec in markets.values():
        up = Path(mspec["universe_file"])
        if up.exists():
            shutil.copy(up, stage / cfg["data_dir"] / up.name)

    with tarfile.open(args.out, "w:gz") as tar:
        tar.add(stage / cfg["data_dir"], arcname=cfg["data_dir"])
    shutil.rmtree(stage)
    size = Path(args.out).stat().st_size / 1e6
    print(f"✓ {args.out}: {kept} ticker (≤{args.years} anni) · {size:.0f} MB")


if __name__ == "__main__":
    main()
    import os
    import sys
    sys.stdout.flush()          # pyarrow lascia thread → uscita forzata a lavoro finito
    sys.stderr.flush()
    os._exit(0)
