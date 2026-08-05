#!/usr/bin/env python3
"""Genera il config del SIMULATORE su titoli VERI.

Perché serve: i segnali di Weinstein sono rari (127 tradeabili su 19.042 titoli
in 24 anni). Dare al gioco titoli veri a caso = partita senza eventi. Qui si
estraggono dal backtest i titoli che PRODUCONO davvero segnali (già filtrati per
liquidità) e si scrive `config_play_real.yaml` con quelli: il gioco ha così una
sequenza densa di decisioni reali, e il confronto umano-vs-bot resta equo (a
entrambi vengono proposti gli stessi segnali).

Uso:  python make_play_config.py [--out config_play_real.yaml] [--max-tickers 120]
Poi:  python play.py --config config_play_real.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.console import setup as _console_setup

_console_setup()

import yaml

from src.backtest import _prepare_backtest, load_universe
from src.config import load_config
from src.data_store import DataStore
from src.providers import make_provider


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_us_liquidity.yaml")
    ap.add_argument("--out", default="config_play_real.yaml")
    ap.add_argument("--max-tickers", type=int, default=120,
                    help="tetto ai titoli nel gioco (l'avvio ricalcola il settimanale)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bt, st_cfg = cfg["backtest"], cfg["stages"]
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    universe = load_universe(cfg)
    print("Scansione dell'universo per trovare i titoli con segnali reali...")
    (_bench, _mkt, _sect, _per_ticker, _tm, all_signals, signal_stats,
     _skipped) = _prepare_backtest(cfg, store, universe, bt["start"], bt["end"],
                                   bt["horizons_weeks"], st_cfg, progress=True)

    # i segnali TRADEABILI (signal_stats è già filtrato per liquidità)
    if signal_stats is None or signal_stats.empty:
        print("Nessun segnale: impossibile costruire il gioco.")
        return
    counts = signal_stats["ticker"].value_counts()
    tickers = list(counts.index[: args.max_tickers])
    n_sig = int(counts.loc[tickers].sum())
    print(f"\n{len(signal_stats)} segnali tradeabili su {counts.size} titoli.")
    print(f"Nel gioco: {len(tickers)} titoli · ~{n_sig} decisioni.")

    spec = cfg["universe"][next(iter(cfg["universe"]))]
    play = {
        "provider": cfg["provider"],
        "data_dir": cfg["data_dir"],
        "output_dir": cfg["output_dir"],
        "universe": {"US": {"mode": "list",
                            "benchmark": spec["benchmark"],
                            "tickers": sorted(tickers)}},
        "backtest": {k: bt[k] for k in ("start", "end", "oos_split", "horizons_weeks")},
        "stages": dict(st_cfg),
        "signal": dict(cfg["signal"]),
        "exits": dict(cfg["exits"]),
        "portfolio": dict(cfg["portfolio"]),
        "simulator": cfg.get("simulator", {"limit_weeks": 4}),
    }
    header = (
        "# ============================================================\n"
        "# SIMULATORE human-vs-bot su TITOLI VERI — generato da make_play_config.py\n"
        "#\n"
        "# I titoli qui elencati sono quelli che producono segnali Weinstein reali\n"
        "# (tradeabili, gia' filtrati per liquidita'). Servono per avere una partita\n"
        "# DENSA: su titoli veri presi a caso i segnali sono rarissimi e la partita\n"
        "# resterebbe vuota. Umano e bot vedono gli STESSI segnali -> confronto equo.\n"
        "#\n"
        "# Avvio:  python play.py --config config_play_real.yaml\n"
        "#         poi apri http://localhost:8000\n"
        "# ============================================================\n"
    )
    out = Path(args.out)
    out.write_text(header + yaml.safe_dump(play, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    print(f"\n✓ Scritto {out}\n  Avvia con:  python play.py --config {out}")


if __name__ == "__main__":
    main()
