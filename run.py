#!/usr/bin/env python3
"""Weinstein Bot — CLI.

Uso:
  python run.py backtest [--config config.yaml] [--out output/report.html]
  python run.py fetch    [--config config.yaml]          # scalda solo la cache

Provider e universo si scelgono in config.yaml. Per EODHD:
  export EODHD_API_KEY=la_tua_chiave
"""
from __future__ import annotations

import argparse
import sys
import time

from src.console import setup as _console_setup

_console_setup()   # UTF-8 su Windows: senza, crasha sulle stampe con → × ✓
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest import load_universe, run_backtest
from src.config import load_config
from src.data_store import DataStore
from src.providers import make_provider
from src.report import build_report


def cmd_fetch(cfg: dict) -> None:
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    bt = cfg["backtest"]
    for market, spec in load_universe(cfg).items():
        for tk in [spec["benchmark"], *spec["tickers"]]:
            df = store.get_with_warmup(tk, bt["start"], bt["end"])
            print(f"[{market}] {tk}: {len(df)} barre  ({df.index.min().date()} → {df.index.max().date()})")


def cmd_backtest(cfg: dict, out: str) -> None:
    t0 = time.time()
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    print(f"Provider: {cfg['provider']} · periodo {cfg['backtest']['start']} → {cfg['backtest']['end']}")
    res = run_backtest(cfg, store, progress=True)
    m = res.metrics
    print(f"\nSegnali: {m['n_signals']} · operazioni: {m['n_trades']}"
          f" · hit rate trade: {m['hit_rate_trades']:.0%}" if m["n_trades"] else "")
    if m.get("n_short_trades"):
        print(f"SHORT: {m['n_short_signals']} segnali · {m['n_short_trades']} operazioni short"
              f" · hit rate {m['short_hit_rate']:.0%}")
    print(f"CAGR bot: {m['strategy']['cagr']:+.1%}  vs benchmark: {m['benchmark']['cagr']:+.1%}")
    print(f"MaxDD bot: {m['strategy']['max_dd']:.1%}  vs benchmark: {m['benchmark']['max_dd']:.1%}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    path = build_report(cfg, res, out)
    print(f"\nReport: {path}  ({time.time() - t0:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Weinstein Bot")
    ap.add_argument("command", choices=["backtest", "fetch"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="output/report.html")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.command == "fetch":
        cmd_fetch(cfg)
    else:
        cmd_backtest(cfg, args.out)


if __name__ == "__main__":
    main()
