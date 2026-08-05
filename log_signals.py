#!/usr/bin/env python3
"""Registra i SEGNALI PIENI settimana per settimana in un log persistente, e ne
traccia la performance DA ALLORA — per decidere un'eventuale entrata tardiva.

Ogni run:
  1. legge i pieni da output/screener.json;
  2. li AGGIUNGE a signals_log.json (dedup su ticker + settimana del segnale: un
     pieno non si registra due volte anche se ricompare per qualche settimana);
  3. AGGIORNA prezzo attuale e % da allora per TUTTI i segnali in log, leggendo
     la cache (adj_close alla settimana del segnale vs adj_close corrente → il
     rendimento totale, dividendi inclusi, su scala di aggiustamento coerente);
  4. genera output/signals.html (pagina separata, linkata dallo screener).

Non costruisce provider né chiama l'API: legge solo file locali (config + cache).
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from src.console import setup as _console_setup

_console_setup()

import pandas as pd

from src.config import load_config


def _perf(root: Path, ticker: str, signal_date: str) -> dict | None:
    """Prezzo corrente e % dalla settimana del segnale, dalla cache giornaliera.
    entry_ref e corrente sono presi dalla STESSA serie (aggiustamento coerente),
    così la % è il rendimento totale reale da allora."""
    safe = ticker.replace("/", "_").replace("\\", "_")
    p = root / f"{safe}.parquet"
    if not p.exists():
        return None
    try:
        s = pd.read_parquet(p, columns=["adj_close"])["adj_close"].dropna()
    except Exception:  # noqa: BLE001
        return None
    if s.empty:
        return None
    at = s.asof(pd.Timestamp(signal_date))     # ultimo prezzo <= settimana segnale
    cur = float(s.iloc[-1])
    pct = None if (at is None or not (at > 0)) else (cur / float(at) - 1.0) * 100.0
    return {"last_price": round(cur, 2),
            "pct_since": round(pct, 1) if pct is not None else None,
            "last_data": s.index[-1].strftime("%Y-%m-%d")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Storico dei segnali pieni + performance da allora")
    ap.add_argument("--config", default="config_us_liquidity.yaml")
    ap.add_argument("--screener", default="output/screener.json")
    ap.add_argument("--log", default="signals_log.json")
    ap.add_argument("--out", default="output/signals.html")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["data_dir"]) / cfg["provider"]
    today = date.today().isoformat()

    logp = Path(args.log)
    log: list[dict] = json.loads(logp.read_text(encoding="utf-8")) if logp.exists() else []
    seen = {(e["ticker"], e["signal_date"]) for e in log}

    # nuovi pieni dallo screener di questa settimana
    data = json.loads(Path(args.screener).read_text(encoding="utf-8"))
    added = 0
    for c in data.get("candidates", []):
        if c.get("kind") != "pieno":
            continue
        key = (c["ticker"], c["date"])
        if key in seen:
            continue
        log.append({
            "ticker": c["ticker"], "market": c["market"], "signal_date": c["date"],
            "entry": c["entry"], "stop": c["stop_bot"], "base_len": c["base_len"],
            "mansfield": c["mansfield"], "vol_ratio": c["vol_ratio"],
            "currency": c.get("currency", ""), "first_logged": today,
        })
        seen.add(key)
        added += 1

    # aggiorna la performance di TUTTI i segnali (anche i vecchi)
    for e in log:
        perf = _perf(root, e["ticker"], e["signal_date"])
        if perf:
            e.update(perf)
        e["weeks_since"] = (pd.Timestamp(today) - pd.Timestamp(e["signal_date"])).days // 7
        e["last_updated"] = today

    log.sort(key=lambda e: e["signal_date"], reverse=True)   # più recenti in cima
    logp.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_html(Path(args.out), log, today)
    print(f"✓ Storico segnali pieni: {len(log)} totali ({added} nuovi) → {args.out}")


def _pct_cell(v) -> str:
    if v is None:
        return "<td class='n'>–</td>"
    cls = "pos" if v > 0 else "neg" if v < 0 else ""
    return f"<td class='n {cls}'>{v:+.1f}%</td>"


def _write_html(path: Path, log: list[dict], today: str) -> None:
    rows = "".join(
        f"<tr><td class='tk'>{e['ticker']}</td>"
        f"<td><span class='mk'>{e['market']}</span></td>"
        f"<td>{e['signal_date']}</td><td class='n'>{e.get('weeks_since','–')}</td>"
        f"<td class='n'>{e['entry']:.2f}</td>"
        f"<td class='n'>{e.get('last_price','–')}</td>"
        + _pct_cell(e.get("pct_since")) +
        f"<td class='n stop'>{e['stop']:.2f}</td>"
        f"<td class='n'>{e['mansfield']:.1f}</td><td>{e['currency']}</td></tr>"
        for e in log
    ) or "<tr><td colspan='10' class='none'>Ancora nessun segnale pieno registrato.</td></tr>"
    html = f"""<!doctype html><meta charset="utf-8"><title>Storico segnali pieni</title>
<style>
 body{{background:#151b21;color:#d7e0e6;font:13px/1.5 -apple-system,Segoe UI,sans-serif;margin:0;padding:24px}}
 h1{{font-size:19px;margin:0 0 4px}} .sub{{color:#7f8c98;font-size:12px;margin-bottom:14px}}
 a{{color:#6fe3a1;text-decoration:none}} a:hover{{text-decoration:underline}}
 table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px}}
 th{{text-align:left;color:#7f8c98;font-weight:600;border-bottom:1px solid #2b353f;padding:7px 8px;white-space:nowrap}}
 td{{border-bottom:1px solid #222c35;padding:7px 8px}} tr:hover td{{background:#1a2128}}
 .n{{text-align:right;font-family:SF Mono,Consolas,monospace}} .tk{{font-weight:600;color:#6fe3a1}}
 .stop{{color:#d6604d}} .pos{{color:#6fe3a1}} .neg{{color:#d6604d}}
 .none{{text-align:center;color:#7f8c98;padding:24px}}
 .mk{{background:#233240;color:#8fb8d8;border-radius:3px;padding:1px 6px;font-size:10px;font-weight:700}}
 .foot{{color:#7f8c98;font-size:11px;margin-top:16px;line-height:1.7}}
</style>
<h1>Storico segnali pieni</h1>
<div class="sub">Aggiornato {today} · {len(log)} segnali registrati · <a href="index.html">← torna allo screener</a></div>
<table>
<tr><th>Ticker</th><th>Mkt</th><th>Settimana segnale</th><th>Sett. fa</th><th>Entry@segnale</th>
<th>Prezzo ora</th><th>% da allora</th><th>Stop</th><th>Mansfield</th><th>Val.</th></tr>
{rows}
</table>
<div class="foot">
Ogni segnale <b>pieno</b> trovato dallo screener viene registrato qui con la settimana in cui è comparso.
<b>% da allora</b> = variazione del prezzo (aggiustato) dalla settimana del segnale a oggi: serve a
valutare un'entrata più tardiva. Non è un portafoglio: qui non si compra né si applicano stop, è un
<i>diario</i> dei segnali. I più recenti in cima.
</div>"""
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
