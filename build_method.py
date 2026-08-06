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
    legend = "".join(
        f"<span class='chip' style='background:{color}'>&nbsp;</span> {lbl}&nbsp;&nbsp;"
        for code, (color, lbl) in STAGE_COLORS.items() if code != 0)

    def row(param, val, why):
        return f"<tr><td class='p'>{param}</td><td class='v'>{val}</td><td class='w'>{why}</td></tr>"

    params = "".join([
        row("MA30", f"{st['ma_weeks']} settimane",
            "la media mobile a 30 settimane che definisce le 4 fasi: sopra e in salita = toro, sotto e in discesa = orso."),
        row("Pendenza MA", f"su {st['slope_lookback']} sett. · piatta se |pendenza| &lt; {st['flat_slope']}",
            "distingue una MA che sale/scende da una laterale."),
        row("Base minima", f"≥ {s['min_base_weeks']} settimane",
            "l'accumulazione (Fase 1) dev'essere matura: mesi, non giorni."),
        row("Profondità base", f"≤ {s['max_base_depth']*100:.0f}%",
            "una base valida è stretta e ordinata, non una montagna russa."),
        row("Volume al breakout", f"≥ {s['volume_ratio_min']}× la media 4 sett.",
            "il breakout vero è confermato da un'esplosione di volume."),
        row("Mansfield RS", f"≥ {s['mansfield_min']:.0f} e in salita da {s['mansfield_rising_weeks']} sett. (finestra 52 sett.)",
            "forza relativa vs mercato: il titolo dev'essere più forte dell'indice."),
        row("Volume nella base", f"contrazione ≤ {s.get('base_volume_max_ratio')} · oppure accumulo OBV ≥ {s.get('base_obv_min')}",
            "nella base il volume si prosciuga (dry-up) o mostra accumulo, poi esplode."),
        row("Anti-notizia", f"scarta se la chiusura è &gt; {s.get('max_breakout_stretch',0)*100:.0f}% sopra la resistenza",
            "un gap enorme da notizia non è una rottura ordinata: spesso ritraccia."),
        row("Requisito A", "la base deve seguire un declino (Fase 4)" if s.get("require_decline") else "disattivato",
            "accumulazione DOPO una discesa, non una pausa dentro un trend."),
        row("Filtro mercato", "Fase 2" if s.get("market_filter") else "off",
            "primo schermo: si compra solo se l'indice è in tendenza rialzista."),
        row("Filtro settore", "Fase 2 (ETF SPDR, US)" if s.get("sector_filter") else "off",
            "secondo schermo: mercato → settore → titolo."),
        row("Stop", f"pivot strutturale, buffer {x.get('pivot_buffer',0.02)*100:.0f}%, sulla CHIUSURA settimanale",
            "sotto l'ultimo minimo relativo; valutato sulla chiusura (metodo settimanale)."),
        row("Esecuzione", x.get("execution", "next_open"),
            "si compra all'apertura della settimana DOPO il segnale (niente look-ahead)."),
    ])

    example = ""
    if ex:
        cap = (f"Esempio reale: <b>{ex['ticker']}</b> · segnale del {ex['signal_date']} · "
               f"entry {ex['entry']} · stop {ex['stop']}. "
               "Sfondo colorato = fasi; tratteggio = resistenza della base; triangolo verde = ENTRY; "
               "linea rossa = STOP; sotto, i volumi con la media a 4 settimane.")
        example = (f"<h2>Un esempio</h2><div class='cap'>{cap}</div>"
                   + (f"<img src='data:image/png;base64,{chart}'>" if chart
                      else "<div class='none'>grafico non disponibile in questa build</div>"))

    html = f"""<!doctype html><meta charset="utf-8"><title>Il metodo · Screener Weinstein</title>
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
<h1>Il metodo</h1>
<div class="sub"><a href="index.html">← screener</a> · <a href="signals.html">follow-up segnali</a></div>

<p>Lo screener applica l'<b>analisi di fase di Stan Weinstein</b> all'intero universo azionario
(USA + Euronext + XETRA). L'idea: comprare quando un titolo <b>rompe al rialzo</b> una base di
accumulazione ed entra in <b>Fase 2</b> (tendenza rialzista), con la conferma di volume e forza relativa.</p>

<h2>Le 4 fasi</h2>
<p>{legend}</p>
<ol>
 <li><b>Fase 1 — Base</b>: dopo un declino, il prezzo si muove lateralmente attorno alla MA30 piatta (accumulazione).</li>
 <li><b>Fase 2 — Avanzata</b>: rottura al rialzo della base, MA30 in salita → è qui che si compra.</li>
 <li><b>Fase 3 — Top</b>: il rialzo si esaurisce, la MA30 si appiattisce (distribuzione).</li>
 <li><b>Fase 4 — Declino</b>: rottura al ribasso, MA30 in discesa → si sta fuori (o short).</li>
</ol>

<h2>Come nasce un segnale</h2>
<p>Alla chiusura settimanale, un candidato è <b>pieno</b> se: c'era una <b>base</b> matura dopo un declino,
la chiusura <b>rompe la resistenza</b> della base ed è sopra la MA30 (non in discesa), il <b>volume</b> esplode,
la <b>forza relativa</b> (Mansfield) è positiva e in salita, e <b>mercato + settore</b> sono in Fase 2.
I "<b>quasi</b>" falliscono uno o due di questi filtri: lo screener li mostra per il tuo giudizio.</p>

<h2>I parametri strutturali (valori attivi)</h2>
<table>{params}</table>

{example}

<div class="note"><b>Onestà sui limiti.</b> Il metodo <b>non batte il mercato</b> in assoluto (resta troppo in
cassa: è market timing). Ma la <b>selezione</b>, misurata contro un "gemello casuale" a pari esposizione e stessi
stop, aggiunge un valore <b>modesto ma positivo</b> (~59° percentile). Il valore vero è la <b>discrezionalità</b>:
pochi titoli scelti col giudizio, non 20.000 meccanicamente → <i>scanner + giudizio umano</i>.</div>

<div class="sub" style="margin-top:20px"><a href="index.html">← torna allo screener</a></div>"""
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
    import os
    import sys
    sys.stdout.flush()          # pyarrow lascia thread → uscita forzata a lavoro finito
    sys.stderr.flush()
    os._exit(0)
