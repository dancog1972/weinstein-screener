#!/usr/bin/env python3
"""Sweep di ESPOSIZIONE: alzando risk_per_trade + max_position_pct la strategia
deploya più capitale per segnale → sta più investita. Misura come cambiano
esposizione media, CAGR e MaxDD. Il gemello resta OFF nello sweep (interessa il
profilo rischio/rendimento della strategia); il percentile si controlla a parte."""
from __future__ import annotations

import sys
from pathlib import Path

from src.console import setup as _console_setup

_console_setup()
sys.path.insert(0, str(Path(__file__).parent))

from src.backtest import run_backtest
from src.config import load_config
from src.data_store import DataStore
from src.providers import make_provider

LEVELS = [
    ("baseline", 0.01, 0.20),
    ("medio",    0.02, 0.40),
    ("alto",     0.04, 0.60),
    ("spinto",   0.06, 0.80),
]


def main() -> None:
    cfg = load_config("config_us_liquidity.yaml")
    cfg["random_twin"] = {"enabled": False}    # sweep: solo profilo della strategia
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))

    print(f"{'livello':10s} {'risk':>5s} {'maxpos':>7s} {'esposiz.':>9s} "
          f"{'CAGR':>7s} {'MaxDD':>8s} {'pos.medie':>10s} {'trade':>6s}")
    print("-" * 70)
    for name, risk, maxpos in LEVELS:
        c = {**cfg, "portfolio": {**cfg["portfolio"],
                                  "risk_per_trade": risk, "max_position_pct": maxpos}}
        res = run_backtest(c, store, progress=False)
        m = res.metrics
        print(f"{name:10s} {risk:>5.0%} {maxpos:>7.0%} "
              f"{m['avg_exposure']*100:>8.1f}% {m['strategy']['cagr']:>+7.1%} "
              f"{m['strategy']['max_dd']:>8.1%} {m['avg_positions']:>10.1f} "
              f"{m['n_trades']:>6d}", flush=True)


if __name__ == "__main__":
    main()
