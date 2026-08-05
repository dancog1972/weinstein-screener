#!/usr/bin/env python3
"""Rigenera i PNG dei 10 casi gia' selezionati (5 migliori / 5 peggiori per
fwd_52w) SENZA rifare il backtest completo: ricostruisce dalla cache solo quei
titoli e riusa il motore vero (detect_signals, forward_returns) e plot_case.

Serve dopo un ritocco al disegno (es. precisione etichette) per non ripagare i
~15 min del backtest sull'intero universo. La lista dei 10 casi viene dal run
completo di make_charts.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.console import setup as _console_setup

_console_setup()
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.config import load_config
from src.data_store import DataStore
from src.indicators import confirmed_pivot_lows, enrich_weekly, to_weekly
from src.providers import make_provider
from src.signals import detect_signals, forward_returns, prepare_ticker
from src.stages import classify_stages
from make_charts import plot_case

# i 10 casi selezionati dal run completo (ordine = rank)
CASES = {
    "best": ["CVSC.US", "KODK-WS-A.US", "KODK.US", "CONN.US", "MSK.US"],
    "worst": ["SMVE.US", "RKLY.US", "EMON.US", "IPKL.US", "TCCC.US"],
}


def main() -> None:
    cfg = load_config("config_us_liquidity.yaml")
    st = cfg["stages"]
    start, end = cfg["backtest"]["start"], cfg["backtest"]["end"]
    horizons = cfg["backtest"]["horizons_weeks"]
    outdir = Path("charts")
    outdir.mkdir(exist_ok=True)

    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    b = store.get_with_warmup(cfg["universe"]["US"]["benchmark"], start, end)
    bw = enrich_weekly(to_weekly(b), None, st["ma_weeks"], st["slope_lookback"])
    mstage = classify_stages(bw, st["flat_slope"])

    for kind, tickers in CASES.items():
        for rank, tk in enumerate(tickers, start=1):
            d = store.get_with_warmup(tk, start, end)
            w = enrich_weekly(to_weekly(d), bw, st["ma_weeks"], st["slope_lookback"])
            df = prepare_ticker(w, st["flat_slope"])
            df["swing_low"] = confirmed_pivot_lows(df, cfg["exits"].get("pivot_k", 2))

            sigs = [s for s in detect_signals(tk, df, mstage, cfg["signal"],
                                              cfg["exits"], st["flat_slope"])
                    if pd.Timestamp(start) <= s.date <= pd.Timestamp(end)]
            # il segnale che ha portato il titolo nel ranking = l'estremo di fwd_52w
            scored = []
            for s in sigs:
                fr = forward_returns(s, df, bw, horizons)
                if pd.notna(fr.get("fwd_52w")):
                    scored.append((fr["fwd_52w"], s))
            if not scored:
                print(f"  [skip] {tk}: nessun segnale con esito a 52w")
                continue
            fwd, s = (max if kind == "best" else min)(scored, key=lambda x: x[0])

            path = plot_case(df, tk, s.date, s.entry, s.stop, fwd,
                             mstage, cfg["signal"], st["flat_slope"], kind, rank, outdir)
            print(f"  ✓ {path.name}")


if __name__ == "__main__":
    main()
