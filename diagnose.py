#!/usr/bin/env python3
"""Diagnostica del funnel: dove muoiono i segnali?

Un backtest che produce zero segnali non è un fallimento: è un'informazione.
Ma senza sapere QUALE filtro li ha uccisi, non puoi farci nulla. Questo script
scompone il funnel condizione per condizione.

  python diagnose.py --config config_us_liquidity.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# UTF-8 su Windows: la console usa cp1252 e crasha sui simboli (→ × ✓ ⚠).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


import pandas as pd

from src.backtest import load_universe
from src.config import load_config
from src.data_store import DataStore
from src.indicators import enrich_weekly, to_weekly
from src.providers import make_provider
from src.stages import base_features, classify_stages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-tickers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    universe = load_universe(cfg)
    st_cfg, sig = cfg["stages"], cfg["signal"]
    start, end = cfg["backtest"]["start"], cfg["backtest"]["end"]

    market = next(iter(universe))
    spec = universe[market]
    b_daily = store.get_with_warmup(spec["benchmark"], start, end)
    bench = enrich_weekly(to_weekly(b_daily), None, st_cfg["ma_weeks"], st_cfg["slope_lookback"])
    mkt_stage = classify_stages(bench, st_cfg["flat_slope"])

    tickers = spec["tickers"]
    if args.max_tickers:
        tickers = tickers[: args.max_tickers]

    # contatori del funnel: ogni settimana di ogni titolo passa (o non passa)
    tot = {k: 0 for k in [
        "settimane_totali", "dati_validi", "liquidita", "stage2_o_base",
        "base_abbastanza_lunga", "base_non_troppo_profonda", "rottura_resistenza",
        "sopra_ma30", "ma_non_in_discesa", "volume_sufficiente",
        "mansfield_positiva", "mansfield_in_salita", "mercato_favorevole",
        "settore_favorevole", "SEGNALE",
    ]}
    per_ticker_signals = {}
    vr_at_breakout: list[float] = []   # volume ratio dei breakout strutturali

    for i, tk in enumerate(tickers, 1):
        try:
            daily = store.get_with_warmup(tk, start, end)
        except Exception:  # noqa: BLE001, S112
            continue
        wk = enrich_weekly(to_weekly(daily), bench, st_cfg["ma_weeks"], st_cfg["slope_lookback"])
        df = wk.loc[start:end].copy()
        if df.empty:
            continue
        wk_stages = classify_stages(wk, st_cfg["flat_slope"])
        df["stage"] = wk_stages.loc[df.index]
        bf = base_features(wk, wk_stages)
        for c in ("resistance", "base_len", "base_depth"):
            if c in bf.columns:
                df[c] = bf[c].loc[df.index]

        n = len(df)
        tot["settimane_totali"] += n
        valid = df["ma"].notna() & df["mansfield"].notna()
        tot["dati_validi"] += int(valid.sum())

        liq = spec.get("liq")
        if liq is not None:
            elig = liq.compute(tk, wk).reindex(df.index).fillna(False)
        else:
            elig = pd.Series(True, index=df.index)
        m = valid & elig
        tot["liquidita"] += int(m.sum())

        res_prev = df["resistance"].shift(1)
        len_prev = df["base_len"].shift(1)
        depth_prev = df["base_depth"].shift(1)

        m &= len_prev.notna()
        tot["stage2_o_base"] += int(m.sum())
        m &= len_prev >= sig["min_base_weeks"]
        tot["base_abbastanza_lunga"] += int(m.sum())
        m &= depth_prev <= sig["max_base_depth"]
        tot["base_non_troppo_profonda"] += int(m.sum())
        m &= df["adj_close"] > res_prev
        tot["rottura_resistenza"] += int(m.sum())
        m &= df["adj_close"] > df["ma"]
        tot["sopra_ma30"] += int(m.sum())
        m &= df["ma_slope"] >= -st_cfg["flat_slope"]
        tot["ma_non_in_discesa"] += int(m.sum())
        # qui il breakout è STRUTTURALMENTE valido: raccolgo i volume ratio
        # per poter tarare la soglia sulla distribuzione, non a intuito
        vr_at_breakout.extend(df.loc[m, "vol_ratio"].dropna().tolist())
        m &= df["vol_ratio"] >= sig["volume_ratio_min"]
        tot["volume_sufficiente"] += int(m.sum())
        m &= df["mansfield"] >= sig["mansfield_min"]
        tot["mansfield_positiva"] += int(m.sum())
        m &= df["mansfield"].diff(sig["mansfield_rising_weeks"]) > 0
        tot["mansfield_in_salita"] += int(m.sum())

        if sig.get("market_filter", True):
            ms = mkt_stage.reindex(df.index).ffill()
            m &= ms == 2
        tot["mercato_favorevole"] += int(m.sum())
        tot["settore_favorevole"] += int(m.sum())   # il settore agisce all'ingresso
        tot["SEGNALE"] += int(m.sum())
        if m.sum():
            per_ticker_signals[tk] = int(m.sum())
        if i % 100 == 0:
            print(f"  ...{i}/{len(tickers)} titoli", flush=True)

    print("\n" + "=" * 72)
    print("FUNNEL DEI SEGNALI — quante settimane-titolo sopravvivono a ogni filtro")
    print("=" * 72)
    prev = None
    for k, v in tot.items():
        if prev is None:
            print(f"  {k:28s} {v:>10,}")
        else:
            drop = prev - v
            pct = drop / prev * 100 if prev else 0
            print(f"  {k:28s} {v:>10,}   (-{drop:,} = -{pct:.1f}%)")
        prev = v

    print("\n" + "=" * 72)
    print(f"SEGNALI TOTALI: {tot['SEGNALE']:,} su {len(tickers)} titoli")
    if per_ticker_signals:
        top = sorted(per_ticker_signals.items(), key=lambda x: -x[1])[:10]
        print(f"titoli con segnali: {len(per_ticker_signals)}")
        print("primi 10:", ", ".join(f"{t}({n})" for t, n in top))
    else:
        print("Nessun segnale. Guarda sopra QUALE riga azzera il funnel:")
        print("  - 'liquidita' → soglie troppo strette per questo universo")
        print("  - 'volume_sufficiente' → volume_ratio_min alto (2.0 è del 1988)")
        print("  - 'mercato_favorevole' → market_filter esclude gran parte del periodo")
    print("=" * 72)

    # ---- distribuzione del volume ratio sui breakout ------------------
    # La soglia 2.0 viene dal 1988. Invece di tararla finché "escono più
    # segnali" (che è ottimizzazione sui dati), guardiamo la DISTRIBUZIONE
    # dei volume ratio nei breakout strutturalmente validi. Una soglia
    # motivata è un percentile della distribuzione, non un numero tondo.
    if vr_at_breakout:
        vr = pd.Series(vr_at_breakout).dropna()
        print("\n" + "=" * 72)
        print("TARATURA DEL VOLUME: distribuzione del volume ratio")
        print(f"nei {len(vr):,} breakout strutturalmente validi")
        print("(base lunga, non profonda, resistenza rotta, sopra MA30, MA non in discesa)")
        print("=" * 72)
        for q in (0.50, 0.70, 0.80, 0.90, 0.95):
            print(f"  {q * 100:4.0f}° percentile: volume_ratio = {vr.quantile(q):.2f}")
        print()
        for soglia in (1.2, 1.5, 1.75, 2.0, 2.5):
            passa = int((vr >= soglia).sum())
            print(f"  soglia {soglia:4.2f} → {passa:5,} breakout passano "
                  f"({passa / len(vr) * 100:4.1f}%)")
        print("\n  La soglia del libro (2.0) è al "
              f"{(vr < 2.0).mean() * 100:.0f}° percentile di questa distribuzione.")
        print("  ATTENZIONE: sceglierla per massimizzare i segnali è overfitting.")
        print("  Una soglia difendibile: il volume del breakout deve stare nella")
        print("  coda alta della distribuzione del titolo, non essere un numero tondo.")
        print("=" * 72)


if __name__ == "__main__":
    main()
