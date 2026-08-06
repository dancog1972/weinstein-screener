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
import os
from datetime import date
from pathlib import Path

from src.console import setup as _console_setup

_console_setup()

import pandas as pd
import requests

from src.config import load_config


def _fetch_watchlist() -> list[dict]:
    """La watchlist dei 'quasi' SEGUITI a mano, dal Worker Cloudflare (env
    WATCH_URL + WATCH_SECRET). Se non configurata → lista vuota (la pagina degrada
    ai soli pieni). Ogni voce ha già ticker/signal_date/entry/stop/... salvati
    dalla pagina al momento del 'segui'."""
    url, secret = os.environ.get("WATCH_URL"), os.environ.get("WATCH_SECRET")
    if not url or not secret:
        return []
    try:
        r = requests.get(f"{url.rstrip('/')}/list", params={"secret": secret}, timeout=15)
        r.raise_for_status()
        return r.json() or []
    except Exception as e:  # noqa: BLE001
        print(f"  [watchlist] non disponibile: {e}")
        return []


def _track(root: Path, e: dict) -> dict | None:
    """Due misure DIVERSE per un segnale:
      - pct_since: prezzo attuale vs settimana del segnale. CORRE sempre, anche se
        lo stop era già scattato (dove sta il titolo ORA → entrata tardiva?).
      - stop_hit / pct_at_stop: se una CHIUSURA SETTIMANALE è scesa sotto lo stop
        iniziale (regola weekly del metodo), con quale performance IN QUEL MOMENTO
        (congelata). È la perdita che avresti preso se fossi entrato e stoppato.
    Confronto per RAPPORTI (return vs rischio) → invariante all'aggiustamento."""
    safe = e["ticker"].replace("/", "_").replace("\\", "_")
    p = root / f"{safe}.parquet"
    if not p.exists():
        return None
    try:
        s = pd.read_parquet(p, columns=["adj_close"])["adj_close"].dropna()
    except Exception:  # noqa: BLE001
        return None
    if s.empty:
        return None
    sig = pd.Timestamp(e["signal_date"])
    at = s.asof(sig)                             # prezzo alla settimana del segnale
    if at is None or not (at > 0):
        return None
    cur = float(s.iloc[-1])
    out = {"last_price": round(cur, 2),
           "pct_since": round((cur / float(at) - 1.0) * 100, 1),
           "last_data": s.index[-1].strftime("%Y-%m-%d"),
           "stop_hit": False, "stop_date": None, "pct_at_stop": None}
    entry, stop = float(e.get("entry", 0)), float(e.get("stop", 0))
    if entry > 0 and stop > 0:
        wk = s.resample("W-FRI").last().dropna()          # chiusure settimanali
        post = wk[wk.index > sig]                          # settimane DOPO il segnale
        if len(post):
            ret = post / float(at) - 1.0                   # return settimana per settimana
            risk = stop / entry - 1.0                       # soglia stop (negativa)
            hit = ret <= risk
            if bool(hit.any()):
                d = post.index[hit.values.argmax()]        # PRIMA settimana sotto lo stop
                out.update(stop_hit=True, stop_date=d.strftime("%Y-%m-%d"),
                           pct_at_stop=round(float(ret.loc[d]) * 100, 1))
    return out


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
        perf = _track(root, e)
        if perf:
            e.update(perf)
        e["weeks_since"] = (pd.Timestamp(today) - pd.Timestamp(e["signal_date"])).days // 7
        e["last_updated"] = today

    log.sort(key=lambda e: e["signal_date"], reverse=True)   # più recenti in cima
    logp.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    # QUASI seguiti a mano (watchlist Cloudflare, se configurata): tracciati come i
    # pieni ma NON committati — sono la selezione VIVA, non uno storico permanente.
    quasi = _fetch_watchlist()
    for w in quasi:
        perf = _track(root, w)
        if perf:
            w.update(perf)
        sd = w.get("signal_date")
        w["weeks_since"] = (pd.Timestamp(today) - pd.Timestamp(sd)).days // 7 if sd else None
    quasi.sort(key=lambda w: w.get("signal_date") or "", reverse=True)

    _write_html(Path(args.out), log, quasi, today)
    print(f"✓ Follow-up: {len(log)} pieni ({added} nuovi) · {len(quasi)} quasi seguiti → {args.out}")


def _pct_cell(v) -> str:
    if v is None:
        return "<td class='n'>–</td>"
    cls = "pos" if v > 0 else "neg" if v < 0 else ""
    return f"<td class='n {cls}'>{v:+.1f}%</td>"


def _num(v, d: int = 2) -> str:
    return f"{v:.{d}f}" if isinstance(v, (int, float)) else "–"


def _row(e: dict) -> str:
    hit = e.get("stop_hit")
    stopcell = (f"<td class='hit'>SÌ · {e.get('stop_date','')}</td>" if hit
                else "<td class='ok2'>no</td>")
    atstop = _pct_cell(e.get("pct_at_stop")) if hit else "<td class='n'>–</td>"
    return (f"<tr><td class='tk'>{e.get('ticker','')}</td>"
            f"<td><span class='mk'>{e.get('market','')}</span></td>"
            f"<td>{e.get('signal_date','')}</td><td class='n'>{e.get('weeks_since','–')}</td>"
            f"<td class='n'>{_num(e.get('entry'))}</td><td class='n stop'>{_num(e.get('stop'))}</td>"
            f"{stopcell}{atstop}"
            f"<td class='n'>{e.get('last_price','–')}</td>{_pct_cell(e.get('pct_since'))}"
            f"<td class='n'>{_num(e.get('mansfield'), 1)}</td><td>{e.get('currency','')}</td></tr>")


_HEADER = ("<tr><th>Ticker</th><th>Mkt</th><th>Settimana segnale</th><th>Sett. fa</th>"
           "<th>Entry@segnale</th><th>Stop</th><th>Stop colpito</th><th>% allo stop</th>"
           "<th>Prezzo ora</th><th>% da allora</th><th>Mansfield</th><th>Val.</th></tr>")


def _table(rows: list[dict], empty: str) -> str:
    body = "".join(_row(e) for e in rows) or f"<tr><td colspan='12' class='none'>{empty}</td></tr>"
    return f"<table>{_HEADER}{body}</table>"


def _write_html(path: Path, pieni: list[dict], quasi: list[dict], today: str) -> None:
    html = f"""<!doctype html><meta charset="utf-8"><title>Follow-up segnali</title>
<style>
 body{{background:#151b21;color:#d7e0e6;font:13px/1.5 -apple-system,Segoe UI,sans-serif;margin:0;padding:24px}}
 h1{{font-size:19px;margin:0 0 4px}} .sub{{color:#7f8c98;font-size:12px;margin-bottom:14px}}
 h2{{font-size:14px;margin:22px 0 4px}} h2 small{{color:#7f8c98;font-weight:400;font-size:11px}}
 .cnt{{background:#233240;border-radius:9px;padding:0 7px;font-size:11px;color:#d7e0e6}}
 a{{color:#6fe3a1;text-decoration:none}} a:hover{{text-decoration:underline}}
 table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:6px}}
 th{{text-align:left;color:#7f8c98;font-weight:600;border-bottom:1px solid #2b353f;padding:7px 8px;white-space:nowrap}}
 td{{border-bottom:1px solid #222c35;padding:7px 8px}} tr:hover td{{background:#1a2128}}
 .n{{text-align:right;font-family:SF Mono,Consolas,monospace}} .tk{{font-weight:600;color:#6fe3a1}}
 .stop{{color:#d6604d}} .pos{{color:#6fe3a1}} .neg{{color:#d6604d}}
 .hit{{color:#d6604d;font-weight:600}} .ok2{{color:#7f8c98}}
 .none{{text-align:center;color:#7f8c98;padding:20px}}
 .mk{{background:#233240;color:#8fb8d8;border-radius:3px;padding:1px 6px;font-size:10px;font-weight:700}}
 .foot{{color:#7f8c98;font-size:11px;margin-top:18px;line-height:1.7}}
</style>
<h1>Follow-up segnali</h1>
<div class="sub">Aggiornato {today} · <a href="index.html">← torna allo screener</a></div>
<h2>Segnali PIENI <span class="cnt">{len(pieni)}</span> <small>automatici, dallo screener</small></h2>
{_table(pieni, "Ancora nessun segnale pieno registrato.")}
<h2>QUASI seguiti <span class="cnt">{len(quasi)}</span> <small>tua selezione — nello screener apri un candidato e clicca ★ segui</small></h2>
{_table(quasi, "Nessun quasi seguito. Nello screener apri un candidato e clicca ★ segui.")}
<div class="foot">
Due misure per riga: <b>Stop colpito</b> / <b>% allo stop</b> = se e con che perdita una chiusura settimanale è
scesa sotto lo stop iniziale (congelata). <b>% da allora</b> = dov'è il titolo ORA rispetto al segnale (corre
sempre, anche dopo lo stop → utile per un'entrata tardiva). Non è un portafoglio: qui non si compra. I PIENI si
accumulano da soli; i QUASI sono la tua watchlist viva (togli il ★ nello screener per smettere di seguirli).
</div>"""
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
    import os
    import sys
    sys.stdout.flush()          # pyarrow lascia thread → uscita forzata a lavoro finito
    sys.stderr.flush()
    os._exit(0)
