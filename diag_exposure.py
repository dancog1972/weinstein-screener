#!/usr/bin/env python3
"""Diagnostico ESPOSIZIONE: la strategia perde contro il gemello perché sceglie
peggio, o perché è meno INVESTITA? Misura la frazione media di capitale investito
e il numero medio di posizioni, per strategia e gemello, sullo stesso universo."""
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


def main() -> None:
    cfg = load_config("config_us_liquidity.yaml")
    # gemello con pochi sorteggi: qui interessa l'esposizione, non il percentile fine
    cfg["random_twin"] = {**cfg.get("random_twin", {}), "enabled": True, "n_sims": 20}
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    res = run_backtest(cfg, store, progress=True)
    m = res.metrics
    tw = m.get("random_twin", {})

    print("\n" + "=" * 60)
    print("ESPOSIZIONE MEDIA (frazione del capitale investita nel tempo)")
    print("-" * 60)
    print(f"  STRATEGIA:  {m['avg_exposure']*100:5.1f}% investito · "
          f"{m['avg_positions']:.1f} posizioni medie · CAGR {m['strategy']['cagr']:+.1%}")
    print(f"  GEMELLO:    {tw.get('avg_exposure', float('nan'))*100:5.1f}% investito · "
          f"{tw.get('avg_positions', float('nan')):.1f} posizioni medie · "
          f"CAGR mediano {tw.get('cagr_p50', float('nan')):+.1%}")
    print("=" * 60)
    print("\nSe il gemello è molto più investito, il suo vantaggio viene")
    print("dall'ESPOSIZIONE, non dalla selezione: in un toro chi è sul mercato")
    print("al 100% batte chi è al 15%, a prescindere da cosa compra.")


if __name__ == "__main__":
    main()
