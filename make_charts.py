#!/usr/bin/env python3
"""Visualizza fasi, basi e segnali del motore Weinstein su titoli reali.

Scopo (richiesta di Daniele): vedere a occhio COME il codice classifica le 4
fasi, dove trova le basi e la loro resistenza, dove scatterebbe il segnale
(con entry e stop) e — soprattutto — quali breakout SCARTA e per quale filtro.

Non riscrive la logica: gira il backtest vero (`run_backtest`), pesca dai
`signal_stats` i 5 segnali col miglior rendimento a 52w e i 5 peggiori (i casi
piu' istruttivi), e per ognuno disegna il titolo usando ESATTAMENTE i
DataFrame gia' arricchiti dal motore (`res.per_ticker`, con stage/base_len/
resistance calcolati da prepare_ticker) e le stesse espressioni di filtro di
`detect_signals`.

Uso:
    python make_charts.py [--config config_us_liquidity.yaml] [--out charts]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.console import setup as _console_setup

_console_setup()   # UTF-8 su Windows

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")            # nessuna finestra: salva solo PNG
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from src.backtest import run_backtest
from src.config import load_config
from src.data_store import DataStore
from src.providers import make_provider

# Colori delle 4 fasi (sfondo). 0 = MA non matura, nessuno sfondo.
STAGE_COLORS = {
    1: ("#d9d9d9", "Stage 1 · Base"),
    2: ("#c6e6c6", "Stage 2 · Advance"),
    3: ("#f5dfa6", "Stage 3 · Top"),
    4: ("#f2c4c4", "Stage 4 · Decline"),
}


def fmt_price(p: float) -> str:
    """Decimali adattivi: i penny/sub-penny stock (best/worst spesso stanno
    proprio li') con `.2f` diventerebbero '0.00', illeggibili."""
    ap = abs(p)
    if ap >= 100:
        return f"{p:.0f}"
    if ap >= 1:
        return f"{p:.2f}"
    if ap >= 0.01:
        return f"{p:.4f}"
    return f"{p:.6f}"


def breakout_checks(df: pd.DataFrame, market_stage: pd.Series | None,
                    sig_cfg: dict, flat_slope: float) -> tuple[pd.Series, dict]:
    """Ricostruisce, condizione per condizione, il filtro di `detect_signals`.

    Le espressioni sono IDENTICHE a quelle in signals.detect_signals: qui pero'
    le teniamo separate per poter dire, di ogni breakout, QUALE filtro l'ha
    bloccato. `core` = evento di breakout grezzo (chiusura sopra la resistenza
    di una base esistente). `checks[label]` = maschera True quando la condizione
    e' RISPETTATA; dove e' False, `label` e' il motivo dello scarto.
    """
    res_prev = df["resistance"].shift(1)
    len_prev = df["base_len"].shift(1)
    depth_prev = df["base_depth"].shift(1)
    mans_rising = df["mansfield"].diff(sig_cfg["mansfield_rising_weeks"]) > 0

    # breakout grezzo: esiste una base (len_prev>=1) e la chiusura la rompe
    core = (len_prev >= 1) & (df["adj_close"] > res_prev)

    checks = {
        "base too short": len_prev >= sig_cfg["min_base_weeks"],
        "base too deep": depth_prev <= sig_cfg["max_base_depth"],
        "below MA30": df["adj_close"] > df["ma"],
        "MA30 falling": df["ma_slope"] > -flat_slope,
        "weak volume": df["vol_ratio"] >= sig_cfg["volume_ratio_min"],
        "Mansfield < min": df["mansfield"] >= sig_cfg["mansfield_min"],
        "Mansfield not rising": mans_rising,
    }
    # filtri opzionali, allineati a detect_signals (altrimenti il grafico
    # segnerebbe come ACCETTATO un breakout che il motore invece scarta)
    if sig_cfg.get("require_decline", True) and "after_decline" in df.columns:
        checks["base not after decline"] = df["after_decline"].shift(1).fillna(False)
    bvr_max = sig_cfg.get("base_volume_max_ratio")
    obv_min = sig_cfg.get("base_obv_min")
    if bvr_max is not None and "base_vol_ratio" in df.columns:
        is_dry = ~(df["base_vol_ratio"].shift(1) > bvr_max)
        if obv_min is not None and "base_obv" in df.columns:
            # passa se dry-up OPPURE accumulo (OBV): scarto solo se NESSUNO dei due
            checks["base no dry-up/accumulation"] = is_dry | (df["base_obv"].shift(1) >= obv_min)
        else:
            checks["base volume not contracted"] = is_dry
    max_stretch = sig_cfg.get("max_breakout_stretch")
    if max_stretch is not None:
        checks["news-driven breakout"] = ~((df["adj_close"] / res_prev - 1.0) > max_stretch)
    if sig_cfg.get("market_filter", True) and market_stage is not None:
        ms = market_stage.reindex(df.index).ffill()
        checks["market not in Stage 2"] = (ms == 2)

    return core.fillna(False), {k: v.fillna(False) for k, v in checks.items()}


def draw_candles(ax, win: pd.DataFrame) -> None:
    """Candele settimanali OHLC (prezzi AGGIUSTATI). Corpo verde se la settimana
    chiude in su, rosso se in giù; stoppino sottile dal minimo al massimo."""
    body_w = 3.4  # larghezza del corpo in giorni (barra settimanale)
    for dt, r in win.iterrows():
        o, h, l, c = r["adj_open"], r["adj_high"], r["adj_low"], r["adj_close"]
        if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(l) and np.isfinite(c)):
            continue
        up = c >= o
        col = "#2ca25f" if up else "#d6604d"
        ax.vlines(dt, l, h, color=col, lw=0.6, zorder=3)     # stoppino
        x = mdates.date2num(dt)
        body_lo, body_hi = (o, c) if up else (c, o)
        height = max(body_hi - body_lo, (h - l) * 1e-3 + 1e-12)
        ax.add_patch(Rectangle((x - body_w / 2, body_lo), body_w, height,
                               facecolor=col, edgecolor=col, lw=0.3, zorder=3))


def contiguous_runs(mask_values: np.ndarray):
    """Elenca gli intervalli [i, j) contigui in cui mask e' True."""
    runs = []
    start = None
    for i, v in enumerate(mask_values):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask_values)))
    return runs


# Etichette brevi per il motivo di USCITA registrato dal portafoglio.
EXIT_LABELS = {
    "stop": "stop iniziale",
    "trailing_stop": "trailing stop",
    "ma_breakdown": "rottura MA30",
    "delisting": "delisting",
    "left_universe": "esce dall'universo",
}


def plot_case(df: pd.DataFrame, ticker: str, sig_date: pd.Timestamp,
              entry: float, stop: float, fwd_52w: float,
              market_stage: pd.Series | None, sig_cfg: dict, flat_slope: float,
              kind: str, rank: int, outdir: Path, trade=None) -> Path:
    """Un grafico per un titolo, centrato sul segnale scelto.

    `trade`: l'operazione realmente eseguita dal portafoglio (o None se il
    segnale non è stato eseguito). Se presente, si disegna punto e tipo di
    USCITA — così si vede cosa ha fatto davvero la strategia, non solo l'esito
    teorico a 52 settimane."""
    core, checks = breakout_checks(df, market_stage, sig_cfg, flat_slope)
    accepted = core.copy()
    for m in checks.values():
        accepted &= m
    rejected = core & ~accepted

    # finestra: ~2.5 anni prima, poi fino a coprire ESITO 52w e USCITA reale
    pos = df.index.get_loc(sig_date)
    lo = max(0, pos - 130)
    end_pos = pos + 62
    if trade is not None and pd.Timestamp(trade.exit_date) in df.index:
        end_pos = max(end_pos, df.index.get_loc(pd.Timestamp(trade.exit_date)) + 8)
    hi = min(len(df), end_pos)
    win = df.iloc[lo:hi]
    idx = win.index

    fig, ax = plt.subplots(figsize=(15, 7.5))

    # --- sfondo: le 4 fasi -------------------------------------------------
    st = win["stage"].to_numpy()
    for code, (color, _label) in STAGE_COLORS.items():
        for a, b in contiguous_runs(st == code):
            left = idx[a]
            right = idx[min(b, len(idx) - 1)]
            ax.axvspan(left, right, color=color, alpha=0.55, linewidth=0, zorder=0)

    # --- prezzo (candele OHLC) e MA30 --------------------------------------
    draw_candles(ax, win)
    ax.plot(idx, win["ma"], color="#2166ac", lw=1.6, label="MA30", zorder=4)
    # le candele sono patch: fisso i limiti Y a mano (min/max della finestra)
    ax.set_ylim(float(win["adj_low"].min()) * 0.96, float(win["adj_high"].max()) * 1.04)

    # --- resistenza delle basi (staircase sui tratti in Fase 1/base) -------
    base_mask = (win["base_len"].to_numpy() > 0)
    res_line = win["resistance"].where(pd.Series(base_mask, index=idx))
    ax.plot(idx, res_line, color="#8c510a", lw=1.3, ls="--",
            label="Resistenza base", zorder=2)

    # --- breakout SCARTATI con motivo -------------------------------------
    rej_win = rejected.reindex(idx).fillna(False)
    rej_dates = idx[rej_win.to_numpy()]
    plotted_reject = False
    for d in rej_dates:
        price = df.loc[d, "adj_close"]
        reasons = [lbl for lbl, m in checks.items() if not bool(m.loc[d])]
        ax.scatter([d], [price], marker="x", s=70, color="#b2182b",
                   zorder=5, linewidths=1.8)
        # etichetta corta col primo motivo (per non intasare)
        if reasons:
            ax.annotate(reasons[0], (d, price), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.5,
                        color="#b2182b", rotation=30)
        plotted_reject = True

    # --- altri segnali ACCETTATI nella finestra (piccoli) ------------------
    acc_win = accepted.reindex(idx).fillna(False)
    for d in idx[acc_win.to_numpy()]:
        if d == sig_date:
            continue
        ax.scatter([d], [df.loc[d, "adj_close"]], marker="^", s=80,
                   color="#1b7837", zorder=5, edgecolors="white", linewidths=0.6)

    # --- il segnale SCELTO: entry, stop, esito 52w -------------------------
    ax.scatter([sig_date], [entry], marker="^", s=230, color="#00441b",
               edgecolors="white", linewidths=1.4, zorder=7, label="Segnale scelto")
    ax.annotate(f"entry {fmt_price(entry)}", (sig_date, entry), textcoords="offset points",
                xytext=(6, 10), fontsize=9, fontweight="bold", color="#00441b")

    # entry e stop come livelli orizzontali attorno al segnale
    x_end = idx[min(len(idx) - 1, idx.get_loc(sig_date) + 55)]
    ax.hlines(entry, sig_date, x_end, color="#00441b", lw=1.0, ls=":", zorder=4)
    ax.hlines(stop, sig_date, x_end, color="#b2182b", lw=1.2, ls=":", zorder=4)
    ax.annotate(f"stop {fmt_price(stop)}  (rischio {(entry-stop)/entry*100:.1f}%)",
                (sig_date, stop), textcoords="offset points",
                xytext=(6, -14), fontsize=8.5, color="#b2182b")

    # esito a 52 settimane
    out_pos = df.index.get_loc(sig_date) + 52
    if out_pos < len(df):
        out_date = df.index[out_pos]
        out_price = df["adj_close"].iloc[out_pos]
        if out_date <= idx[-1]:
            ax.scatter([out_date], [out_price], marker="o", s=90,
                       color="#762a83", zorder=6, edgecolors="white", linewidths=0.8)
            ax.annotate(f"+52w: {fwd_52w*100:+.0f}%", (out_date, out_price),
                        textcoords="offset points", xytext=(6, 8),
                        fontsize=9, fontweight="bold", color="#762a83")

    # --- USCITA reale del trade (cosa ha fatto DAVVERO la strategia) --------
    if trade is not None and pd.Timestamp(trade.exit_date) in df.index:
        ex_date = pd.Timestamp(trade.exit_date)
        ex_px = float(trade.exit_price)
        reason = EXIT_LABELS.get(trade.reason, trade.reason)
        # linea entry→uscita: verde se il trade ha guadagnato, rossa se ha perso
        won = trade.exit_price >= trade.entry_price
        col = "#1b7837" if won else "#b2182b"
        ax.plot([sig_date, ex_date], [entry, ex_px], color=col, lw=1.0,
                ls="-", alpha=0.5, zorder=3)
        ax.scatter([ex_date], [ex_px], marker="s", s=130, color="#d94801",
                   edgecolors="white", linewidths=1.2, zorder=8)
        ax.annotate(f"USCITA: {reason}\n{fmt_price(ex_px)}  ({trade.ret*100:+.0f}%)",
                    (ex_date, ex_px), textcoords="offset points", xytext=(6, -22),
                    fontsize=8.5, fontweight="bold", color="#8a3800")

    # --- cornice, legenda, titolo -----------------------------------------
    tag = "MIGLIORE" if kind == "best" else "PEGGIORE"
    ax.set_title(f"{tag} #{rank} · {ticker} · segnale {sig_date.date()} · "
                 f"rendimento a 52w = {fwd_52w*100:+.1f}%",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Prezzo aggiustato")
    ax.grid(True, axis="y", alpha=0.25)
    ax.margins(x=0.01)

    handles = [
        Patch(facecolor="#2ca25f", label="Settimana su"),
        Patch(facecolor="#d6604d", label="Settimana giù"),
        Line2D([], [], color="#2166ac", lw=1.6, label="MA30"),
        Line2D([], [], color="#8c510a", lw=1.3, ls="--", label="Resistenza base"),
        Line2D([], [], marker="^", color="#00441b", lw=0, markersize=11,
               markeredgecolor="white", label="Segnale scelto"),
        Line2D([], [], marker="^", color="#1b7837", lw=0, markersize=8,
               markeredgecolor="white", label="Altro segnale accettato"),
        Line2D([], [], marker="x", color="#b2182b", lw=0, markersize=8,
               label="Breakout scartato (col motivo)"),
        Line2D([], [], marker="o", color="#762a83", lw=0, markersize=8,
               markeredgecolor="white", label="Prezzo a +52w"),
        Line2D([], [], marker="s", color="#d94801", lw=0, markersize=9,
               markeredgecolor="white", label="Uscita reale del trade"),
    ]
    handles += [Patch(facecolor=c, alpha=0.55, label=lbl)
                for _, (c, lbl) in STAGE_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=2, framealpha=0.9)

    fig.tight_layout()
    safe = ticker.replace(".", "_")
    fname = outdir / f"{kind}_{rank}_{safe}_{fwd_52w*100:+.0f}pct.png"
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    return fname


def main() -> None:
    ap = argparse.ArgumentParser(description="Grafici fasi/segnali Weinstein")
    ap.add_argument("--config", default="config_us_liquidity.yaml")
    ap.add_argument("--out", default="charts")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["random_twin"] = {"enabled": False}   # inutile qui, costoso
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    print(f"Backtest completo su {cfg['provider']} "
          f"({cfg['backtest']['start']} → {cfg['backtest']['end']}) · gemello OFF")
    res = run_backtest(cfg, store, progress=True)

    ss = res.signal_stats
    if ss is None or ss.empty or "fwd_52w" not in ss:
        print("Nessun segnale con fwd_52w: impossibile scegliere i casi.")
        return
    ss = ss.dropna(subset=["fwd_52w"]).copy()
    ss = ss.sort_values("fwd_52w")
    worst = ss.head(5)
    best = ss.tail(5).iloc[::-1]

    m = res.metrics
    print(f"\nCAGR {m['strategy']['cagr']:+.1%} · MaxDD {m['strategy']['max_dd']:.1%} · "
          f"hit {m['hit_rate_trades']:.0%} · {m['n_trades']} trade "
          f"(bench CAGR {m['benchmark']['cagr']:+.1%})")
    print(f"{len(ss)} segnali con esito a 52w noto.")
    print(f"Migliori 5 (fwd_52w): "
          f"{', '.join(f'{r.ticker} {r.fwd_52w*100:+.0f}%' for r in best.itertuples())}")
    print(f"Peggiori 5 (fwd_52w): "
          f"{', '.join(f'{r.ticker} {r.fwd_52w*100:+.0f}%' for r in worst.itertuples())}")

    sig_cfg = cfg["signal"]
    flat_slope = cfg["stages"]["flat_slope"]
    market_stage = res.market_stage

    # trade per ticker, per abbinare a ciascun segnale l'operazione reale.
    # L'esecuzione è a next_open: il trade nasce alla PRIMA settimana dopo il
    # segnale. Abbino su ticker + entry_date == quella settimana.
    trades_by_tk: dict[str, list] = {}
    for t in res.trades:
        trades_by_tk.setdefault(t.ticker, []).append(t)

    def match_trade(tk: str, df: pd.DataFrame, sig_date: pd.Timestamp):
        after = df.index[df.index > sig_date]
        if len(after) == 0:
            return None
        exec_date = after[0]
        for t in trades_by_tk.get(tk, []):
            if pd.Timestamp(t.entry_date) == exec_date:
                return t
        return None

    made = []
    for kind, block in [("best", best), ("worst", worst)]:
        for rank, row in enumerate(block.itertuples(), start=1):
            tk = row.ticker
            df = res.per_ticker.get(tk)
            if df is None or pd.Timestamp(row.date) not in df.index:
                print(f"  [skip] {tk}: dati non disponibili per il grafico")
                continue
            trade = match_trade(tk, df, pd.Timestamp(row.date))
            note = f"uscita={trade.reason}" if trade else "NON eseguito"
            path = plot_case(df, tk, pd.Timestamp(row.date), float(row.entry),
                             float(row.stop), float(row.fwd_52w),
                             market_stage, sig_cfg, flat_slope, kind, rank, outdir,
                             trade=trade)
            print(f"  ✓ {path.name}  ({note})")
            made.append(path)

    print(f"\n{len(made)} grafici salvati in {outdir}/")


if __name__ == "__main__":
    main()
