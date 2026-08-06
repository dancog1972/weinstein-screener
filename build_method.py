#!/usr/bin/env python3
"""Genera metodo.html: la pagina che spiega il METODO usato per i segnali —
le 4 fasi di Weinstein, i parametri strutturali (coi VALORI presi dal config,
così la pagina non diverge mai dal codice) e un esempio grafico su un segnale reale.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from src.console import setup as _console_setup

_console_setup()

import pandas as pd
import yaml

from src.config import load_config
from src.data_store import DataStore
from src.indicators import enrich_weekly, to_weekly
from src.providers import make_provider
from src.signals import prepare_ticker
from make_charts import STAGE_COLORS
from screener import chart_b64


def _example_chart(cfg: dict, markets: dict, ex: dict) -> str | None:
    """Grafico annotato di un segnale reale (preso dal signals_log): fasi colorate,
    base + resistenza, entry e stop — lo stesso disegno delle schede dello screener."""
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    st = cfg["stages"]
    today = pd.Timestamp(date.today())
    start = (today - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    mspec = markets[ex["market"]]
    bench = enrich_weekly(to_weekly(store.get_with_warmup(mspec["benchmark"], start, end)),
                          None, st["ma_weeks"], st["slope_lookback"])
    df = prepare_ticker(
        enrich_weekly(to_weekly(store.get_with_warmup(ex["ticker"], start, end)),
                      bench, st["ma_weeks"], st["slope_lookback"]),
        st["flat_slope"])
    sig = SimpleNamespace(ticker=ex["ticker"], date=pd.Timestamp(ex["signal_date"]), entry=float(ex["entry"]))
    return chart_b64(df, sig, float(ex["stop"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera metodo.html (spiegazione + parametri + esempio)")
    ap.add_argument("--config", default="config_us_liquidity.yaml")
    ap.add_argument("--markets", default="config_screener_markets.yaml")
    ap.add_argument("--log", default="signals_log.json")
    ap.add_argument("--out", default="output/metodo.html")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.markets, encoding="utf-8") as fh:
        markets = yaml.safe_load(fh)["markets"]

    chart, ex = None, None
    logp = Path(args.log)
    if logp.exists():
        log = json.loads(logp.read_text(encoding="utf-8"))
        if log:
            ex = log[0]                      # un pieno reale come esempio
            try:
                chart = _example_chart(cfg, markets, ex)
            except Exception as e:  # noqa: BLE001
                print(f"  [esempio] grafico non disponibile: {e}")
    _write(Path(args.out), cfg, ex, chart)
    print(f"✓ Pagina metodo → {args.out}")


def _write(path: Path, cfg: dict, ex: dict | None, chart: str | None) -> None:
    s, st, x = cfg["signal"], cfg["stages"], cfg["exits"]
    pf = cfg["portfolio"]
    legend = "".join(
        f"<span class='chip' style='background:{color}'>&nbsp;</span> {lbl}&nbsp;&nbsp;"
        for code, (color, lbl) in STAGE_COLORS.items() if code != 0)

    def row(param, val, why):
        return f"<tr><td class='p'>{param}</td><td class='v'>{val}</td><td class='w'>{why}</td></tr>"

    params = "".join([
        row("MA30", f"{st['ma_weeks']} weeks",
            "the 30-week moving average that defines the 4 stages: above and rising = bull, below and falling = bear."),
        row("MA slope", f"over {st['slope_lookback']} w · flat if |slope| &lt; {st['flat_slope']}",
            "tells a rising/falling MA from a sideways one."),
        row("Min base", f"≥ {s['min_base_weeks']} weeks",
            "the accumulation (Stage 1) must be mature: months, not days."),
        row("Base depth", f"≤ {s['max_base_depth']*100:.0f}%",
            "a valid base is tight and orderly, not a roller-coaster."),
        row("Breakout volume", f"≥ {s['volume_ratio_min']}× the 4-week average",
            "a real breakout is confirmed by a volume surge."),
        row("Mansfield RS", f"≥ {s['mansfield_min']:.0f} and rising for {s['mansfield_rising_weeks']} w (52-week window)",
            "relative strength vs market: the stock must be stronger than the index."),
        row("Base volume", f"contraction ≤ {s.get('base_volume_max_ratio')} · or OBV accumulation ≥ {s.get('base_obv_min')}",
            "in the base volume dries up (dry-up) or shows accumulation, then explodes."),
        row("Anti-news", f"rejected if the close is &gt; {s.get('max_breakout_stretch',0)*100:.0f}% above resistance",
            "a huge news gap isn't an orderly break: it often retraces."),
        row("Requirement A", "the base must follow a decline (Stage 4)" if s.get("require_decline") else "off",
            "accumulation AFTER a decline, not a pause inside a trend."),
        row("Market filter", "Stage 2" if s.get("market_filter") else "off",
            "first screen: buy only if the index is in an uptrend."),
        row("Sector filter", "Stage 2 (SPDR ETFs, US)" if s.get("sector_filter") else "off",
            "second screen: market → sector → stock."),
        row("Stop", f"structural pivot, {x.get('pivot_buffer',0.02)*100:.0f}% buffer, on the weekly CLOSE",
            "below the last relative low; evaluated on the close (weekly method)."),
        row("Execution", x.get("execution", "next_open"),
            "buy at the open of the week AFTER the signal (no look-ahead)."),
    ])

    example = ""
    if ex:
        cap = (f"Real example: <b>{ex['ticker']}</b> · signal on {ex['signal_date']} · "
               f"entry {ex['entry']} · stop {ex['stop']}. "
               "Colored background = stages; dashed line = base resistance; green triangle = ENTRY; "
               "red line = STOP; below, volume with the 4-week average.")
        example = (f"<h2>An example</h2><div class='cap'>{cap}</div>"
                   + (f"<img src='data:image/png;base64,{chart}'>" if chart
                      else "<div class='none'>chart not available in this build</div>"))

    html = f"""<!doctype html><meta charset="utf-8"><title>The method · Weinstein Screener</title>
<style>
 body{{background:#151b21;color:#d7e0e6;font:14px/1.65 -apple-system,Segoe UI,sans-serif;margin:0 auto;padding:26px;max-width:900px}}
 h1{{font-size:22px;margin:0 0 6px}} h2{{font-size:16px;margin:26px 0 8px;color:#6fe3a1}}
 .sub{{color:#7f8c98;font-size:12px;margin-bottom:18px}} a{{color:#6fe3a1;text-decoration:none}} a:hover{{text-decoration:underline}}
 p{{margin:8px 0}} b{{color:#eaf2f6}}
 .chip{{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:middle;opacity:.7}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
 td{{border-bottom:1px solid #222c35;padding:8px 10px;vertical-align:top}}
 .p{{font-weight:600;color:#eaf2f6;white-space:nowrap}} .v{{font-family:SF Mono,Consolas,monospace;color:#6fe3a1;white-space:nowrap}}
 .w{{color:#a9b6c0}} img{{width:100%;height:auto;border-radius:6px;border:1px solid #2b353f;margin-top:6px}}
 .cap{{color:#7f8c98;font-size:12px}} .none{{color:#7f8c98;padding:20px;text-align:center}}
 .note{{background:#1a2128;border-left:3px solid #e0a458;padding:10px 14px;border-radius:4px;margin:16px 0;color:#c9d3da;font-size:13px}}
 ol{{padding-left:20px}} li{{margin:4px 0}}
</style>
<h1>The method</h1>
<div class="sub"><a href="index.html">← screener</a> · <a href="signals.html">signals follow-up</a></div>

<p>The screener applies <b>Stan Weinstein's stage analysis</b> to the whole stock universe
(US + Euronext + XETRA). The idea: buy when a stock <b>breaks out</b> of an accumulation base
and enters <b>Stage 2</b> (uptrend), confirmed by volume and relative strength.</p>

<h2>The 4 stages</h2>
<p>{legend}</p>
<ol>
 <li><b>Stage 1 — Base</b>: after a decline, price moves sideways around a flat MA30 (accumulation).</li>
 <li><b>Stage 2 — Advance</b>: breakout above the base, MA30 rising → this is where you buy.</li>
 <li><b>Stage 3 — Top</b>: the advance stalls, the MA30 flattens (distribution).</li>
 <li><b>Stage 4 — Decline</b>: breakdown, MA30 falling → stay out (or short).</li>
</ol>

<h2>How a signal is born</h2>
<p>At the weekly close, a candidate is <b>full</b> if: there was a mature <b>base</b> after a decline,
the close <b>breaks the resistance</b> of the base and is above the MA30 (not falling), <b>volume</b> surges,
<b>relative strength</b> (Mansfield) is positive and rising, and <b>market + sector</b> are in Stage 2.
"<b>Near</b>" candidates fail one or two of these filters: the screener shows them for your judgment.</p>

<h2>Exit strategy</h2>
<p>Once in, the position is managed with the <b>validated exit</b> (the same one used in the backtest, so it can't
diverge from the method). The trade exits on whichever of these triggers first:</p>
<ol>
 <li><b>Initial stop</b> — below the last confirmed relative low (structural pivot), {x.get('pivot_buffer',0.02)*100:.0f}% buffer, evaluated on the <b>weekly close</b>.</li>
 <li><b>Trailing stop on the MA30</b> — each week the stop rises to <b>MA30 − {x.get('ma_trail_atr_mult',2)}×ATR</b> (only upward), so a winner's stop climbs with the trend and locks in gains (Weinstein's "hold through Stage 2").</li>
 <li><b>MA30 breakdown</b> — {'on' if x.get('ma_breakdown') else 'off'}: exit when the weekly close drops below a falling MA30.</li>
</ol>
<p>On the <a href="signals.html">follow-up</a> page each signal shows its real exit (date, price, reason) and the
realized return, next to where the stock is now (so you can see a recovery after the stop, or a late entry).</p>

<h2>Risk</h2>
<p>The <b>Risk</b> shown on the pages is the distance from entry to the initial stop:
<b>Risk% = (entry − stop) / entry</b> — how much you would lose if the initial stop is hit. It also drives
<b>position sizing</b>: each trade risks a fixed <b>{pf['risk_per_trade']*100:.0f}% of capital</b>
(of {pf['initial_capital']:,}), so <i>shares = risk_amount / (entry − stop)</i>. A wide stop → a smaller
position; a tight stop → a larger one (capped at {pf['max_position_pct']*100:.0f}% of capital per position).
The share/position numbers are illustrative — recompute them on your own capital.</p>

<h2>Structural parameters (active values)</h2>
<table>{params}</table>

{example}

<div class="note"><b>Honest about the limits.</b> The method <b>does not beat the market</b> outright (it stays too
much in cash: it's market timing). But the <b>selection</b>, measured against a "random twin" at equal exposure and
same stops, adds <b>modest but positive</b> value (~59th percentile). The real value is <b>discretion</b>:
a few stocks chosen with judgment, not 20,000 mechanically → <i>scanner + human judgment</i>.</div>

<div class="sub" style="margin-top:20px"><a href="index.html">← back to screener</a></div>"""
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
    import os
    import sys
    sys.stdout.flush()          # pyarrow lascia thread → uscita forzata a lavoro finito
    sys.stderr.flush()
    os._exit(0)
