"""Dashboard HTML standalone (approccio A): file autonomo, funziona offline.

Contiene: metriche strategia vs benchmark (con split IS/OOS), curva equity,
qualità dei segnali per orizzonte, tabella segnali, grafico di un titolo
d'esempio con la banda delle fasi, e il pannello config: la configurazione
usata per il run è incorporata nel file e riesportabile modificata.
Stile: quaderno di ricerca quantitativa — carta, inchiostro, dati in mono.
"""
from __future__ import annotations

import datetime as dt
import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .backtest import BacktestResult
from .stages import STAGE_NAMES

INK = "#1d2733"        # inchiostro
PAPER = "#fbfaf7"      # carta
GRID = "#e4e0d6"
STRAT = "#0e5a8a"      # blu di prussia — strategia
BENCH = "#8a7f6a"      # seppia — benchmark
STAGE_COLORS = {1: "#d9c97f", 2: "#7fb069", 3: "#e0a458", 4: "#c1666b"}

_PLOTLY_LAYOUT = dict(
    paper_bgcolor=PAPER, plot_bgcolor=PAPER,
    font=dict(family="Georgia, 'Times New Roman', serif", color=INK, size=13),
    margin=dict(l=55, r=25, t=40, b=40),
    xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
    legend=dict(orientation="h", y=1.08, x=0),
)


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.1f}%"


def _fmt_num(x: float | None, nd: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ----------------------------------------------------------------------
def _equity_fig(res: BacktestResult, oos_split: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=res.equity_curve.index, y=res.equity_curve.values,
                    name="Bot Weinstein", line=dict(color=STRAT, width=2))
    fig.add_scatter(x=res.bench_curve.index, y=res.bench_curve.values,
                    name="Benchmark (buy & hold)", line=dict(color=BENCH, width=1.6, dash="dot"))
    fig.add_vline(x=pd.Timestamp(oos_split), line=dict(color=INK, width=1, dash="dash"))
    fig.add_annotation(x=pd.Timestamp(oos_split), y=1, yref="paper", showarrow=False,
                       text="in-sample ◂ ▸ out-of-sample", font=dict(size=11), yshift=12)
    fig.update_layout(title="Curva equity (scala log)", yaxis_type="log",
                      height=420, **_PLOTLY_LAYOUT)
    return fig


def _ticker_fig(ticker: str, df: pd.DataFrame, signals: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["ma"])
    fig = go.Figure()
    # banda delle fasi: la firma visiva del report
    st = d["stage"].to_numpy()
    dates = d.index
    run_start = 0
    for i in range(1, len(st) + 1):
        if i == len(st) or st[i] != st[run_start]:
            s = int(st[run_start])
            if s in STAGE_COLORS:
                fig.add_vrect(x0=dates[run_start], x1=dates[min(i, len(st) - 1)],
                              fillcolor=STAGE_COLORS[s], opacity=0.14, line_width=0)
            run_start = i
    fig.add_scatter(x=d.index, y=d["adj_close"], name="Prezzo (adj)",
                    line=dict(color=INK, width=1.3))
    fig.add_scatter(x=d.index, y=d["ma"], name="MA 30 settimane",
                    line=dict(color=STRAT, width=1.6))
    mine = (signals[signals["ticker"] == ticker]
            if len(signals) and "ticker" in signals.columns else signals)
    if len(mine):
        fig.add_scatter(x=mine["date"], y=mine["entry"], mode="markers",
                        name="Segnale buy", marker=dict(symbol="triangle-up",
                        size=11, color=STAGE_COLORS[2], line=dict(color=INK, width=1)))
    fig.update_layout(title=f"{ticker} — fasi e segnali (settimanale)",
                      height=420, **_PLOTLY_LAYOUT)
    return fig


def _mansfield_fig(ticker: str, df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["mansfield"])
    fig = go.Figure()
    fig.add_scatter(x=d.index, y=d["mansfield"], name="Mansfield RS",
                    line=dict(color=BENCH, width=1.3))
    fig.add_hline(y=0, line=dict(color=INK, width=1))
    fig.update_layout(title=f"{ticker} — forza relativa Mansfield",
                      height=240, **_PLOTLY_LAYOUT)
    return fig


# ----------------------------------------------------------------------
def _metrics_table(m: dict) -> str:
    rows = [
        ("CAGR", m["strategy"]["cagr"], m["benchmark"]["cagr"], True),
        ("Max drawdown", m["strategy"]["max_dd"], m["benchmark"]["max_dd"], True),
        ("Sharpe (settimanale ann.)", m["strategy"]["sharpe"], m["benchmark"]["sharpe"], False),
        ("CAGR in-sample", m["strategy_is"]["cagr"], m["benchmark_is"]["cagr"], True),
        ("CAGR out-of-sample", m["strategy_oos"]["cagr"], m["benchmark_oos"]["cagr"], True),
    ]
    body = "".join(
        f"<tr><td>{name}</td><td class='num'>{_fmt_pct(a) if pct else _fmt_num(a)}</td>"
        f"<td class='num'>{_fmt_pct(b) if pct else _fmt_num(b)}</td></tr>"
        for name, a, b, pct in rows)
    return (f"<table><thead><tr><th></th><th>Bot Weinstein</th><th>Benchmark</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def _signal_quality_table(m: dict, horizons: list[int]) -> str:
    ss = m.get("signal_summary", {})
    if not ss:
        return "<p>Nessun segnale nel periodo.</p>"
    head = "".join(f"<th>hit {h}w</th><th>exc. medio {h}w</th>" for h in horizons)
    body = ""
    for seg, label in [("all", "Tutti"), ("is", "In-sample"), ("oos", "Out-of-sample")]:
        r = ss.get(seg, {})
        cells = "".join(
            f"<td class='num'>{_fmt_pct(r.get(f'hit_{h}w'))}</td>"
            f"<td class='num'>{_fmt_pct(r.get(f'avg_exc_{h}w'))}</td>" for h in horizons)
        body += f"<tr><td>{label} (n={r.get('n', 0)})</td>{cells}</tr>"
    return (f"<table><thead><tr><th>Segmento</th>{head}</tr></thead><tbody>{body}</tbody></table>"
            "<p class='note'>hit = quota di segnali con rendimento in eccesso positivo sul "
            "benchmark all'orizzonte indicato; exc. medio = rendimento medio in eccesso.</p>")


def _signals_table(sig_stats: pd.DataFrame, horizons: list[int], limit: int = 400) -> str:
    if not len(sig_stats):
        return "<p>Nessun segnale.</p>"
    df = sig_stats.sort_values("date", ascending=False).head(limit)
    hcols = [f"exc_{h}w" for h in horizons]
    head = ("<tr><th>Data</th><th>Ticker</th><th>Entry</th><th>Stop</th><th>Base (sett.)</th>"
            "<th>Vol ratio</th><th>Mansfield</th>"
            + "".join(f"<th>exc {h}w</th>" for h in horizons) + "</tr>")
    rows = []
    for _, r in df.iterrows():
        cells = "".join(f"<td class='num'>{_fmt_pct(r[c])}</td>" for c in hcols)
        rows.append(
            f"<tr><td>{r['date'].date()}</td><td>{html.escape(str(r['ticker']))}</td>"
            f"<td class='num'>{r['entry']:.2f}</td><td class='num'>{r['stop']:.2f}</td>"
            f"<td class='num'>{int(r['base_len'])}</td><td class='num'>{r['vol_ratio']:.1f}×</td>"
            f"<td class='num'>{r['mansfield']:+.1f}</td>{cells}</tr>")
    return f"<table id='sigtable'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


# ----------------------------------------------------------------------
def _vol_note(m: dict, soglie: dict) -> str:
    """Dichiara la scelta di misurare la volatilità invece della capitalizzazione,
    e riporta se il filtro aggiunge davvero selettività."""
    vmax = soglie.get("volatilita_max")
    if not vmax:
        return ("<p class='warn'>Filtro di <b>volatilità disattivato</b>. Weinstein "
                "preferisce titoli con 'variazioni contenute': senza questo criterio "
                "l'universo può includere small-cap nervose, prone a falsi breakout.</p>")
    vc = m.get("volatility_check", {})
    corr = vc.get("correlazione_turnover_volatilita")
    interp = vc.get("interpretazione", "")
    quota = vc.get("quota_esclusa_solo_da_volatilita_pct")
    corr_txt = (f"Correlazione turnover↔volatilità: <b>{corr:+.3f}</b>. "
                if corr is not None else "")
    return (f"<p class='ok'>✓ <b>Volatilità</b> ≤ {vmax:.0%} (ATR settimanale sul prezzo). "
            f"Misura DIRETTAMENTE ciò che la capitalizzazione approssimava: Weinstein "
            f"vuole 'variazioni contenute', e la volatilità si legge dai prezzi senza "
            f"dati fondamentali a pagamento — e senza il look-ahead che si avrebbe "
            f"applicando la capitalizzazione odierna alla storia passata. {corr_txt}"
            + (f"Il criterio esclude un ulteriore <b>{quota}%</b> delle settimane che "
               f"passavano il turnover: <i>{interp}</i>." if quota is not None else "")
            + "</p>")


def build_report(cfg: dict, res: BacktestResult, out_path: str) -> str:
    m = res.metrics
    horizons = cfg["backtest"]["horizons_weeks"]
    # titolo d'esempio: quello con più segnali
    example = (res.signal_stats["ticker"].value_counts().idxmax()
               if len(res.signal_stats) else next(iter(res.per_ticker)))
    fig_equity = _equity_fig(res, cfg["backtest"]["oos_split"])
    fig_ticker = _ticker_fig(example, res.per_ticker[example], res.signal_stats)
    fig_rs = _mansfield_fig(example, res.per_ticker[example])

    plots = (fig_equity.to_html(full_html=False, include_plotlyjs="inline", config={"displaylogo": False}),
             fig_ticker.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False}),
             fig_rs.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False}))

    stage_legend = " · ".join(
        f"<span class='chip' style='background:{c}22;border-color:{c}'>{n}. {STAGE_NAMES[n]}</span>"
        for n, c in STAGE_COLORS.items())

    cfg_yaml = html.escape(cfg.get("_raw_yaml", ""))
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    provider = html.escape(cfg["provider"])
    caveat_synth = ("<p class='warn'>⚠ Questo run usa il provider <b>synthetic</b>: dati generati, "
                    "utili solo a validare il motore. I numeri non dicono nulla sui mercati reali.</p>"
                    if cfg["provider"] == "synthetic" else "")

    # dichiarazione della modalità universo: cruciale per interpretare i numeri
    modes = m.get("universe_mode", {})
    uni_note = ""
    if modes:
        n = m.get("universe_size", 0)
        skipped = m.get("signals_skipped_pit", 0)
        if "liquidity" in modes.values():
            ls = m.get("liquidity_stats", {})
            soglie = ls.get("soglie", {})
            sk_liq = m.get("signals_skipped_liq", 0)
            sk_sec = m.get("signals_skipped_sector", 0)
            sec_on = m.get("sector_filter_active", False)
            uni_note = (
                f"<p class='ok'>✓ <b>Universo per liquidità</b> su {n} titoli valutati "
                f"(prezzo ≥ ${soglie.get('prezzo_min','?')}, volume ≥ "
                f"{soglie.get('volume_min','?'):,.0f}, turnover ≥ "
                f"${soglie.get('turnover_min','?'):,.0f}). L'idoneità di ogni settimana è "
                f"giudicata sulle settimane <i>precedenti</i> (medie con shift): nessun "
                f"look-ahead. Un titolo entra quando diventa liquido ed esce quando smette, "
                f"e i delistati escono quando muoiono: <b>il survivorship bias è eliminato "
                f"per costruzione</b>. "
                f"Membri per settimana: {ls.get('membri_per_settimana_medio','?')} in media "
                f"(min {ls.get('membri_per_settimana_min','?')}, "
                f"max {ls.get('membri_per_settimana_max','?')}). "
                f"Segnali scartati: {sk_liq} per liquidità"
                + (f", {sk_sec} per settore debole (triple screen attivo)." if sec_on
                   else ". Filtro settoriale <b>inattivo</b>.")
                + "</p>"
                + _vol_note(m, soglie)
                + "<p class='warn'>Bias residui dichiarati (nessuno gonfia i "
                  "risultati a favore; semmai li rende più severi): "
                  "<b>(1)</b> la mappa ticker→settore è statica, cioè la "
                  "classificazione odierna applicata alla storia passata. "
                  "<b>(2)</b> il delisting chiude la posizione all'ultimo prezzo "
                  "noto: realistico per fusioni/acquisizioni, ottimistico per i "
                  "fallimenti (il valore reale di liquidazione può essere più "
                  "basso). Il piano dati non fornisce il motivo del delisting, "
                  "quindi la stima è un limite superiore su quei casi. "
                  "<b>(3)</b> l'esecuzione avviene all'apertura della settimana "
                  "successiva al segnale (non alla chiusura del breakout stesso): "
                  "questo È già la scelta onesta, dichiarata qui per trasparenza.</p>")
        elif "index_pit" in modes.values():
            uni_note = (f"<p class='ok'>✓ <b>Universo point-in-time</b> su {n} titoli storici: "
                        f"un titolo è comprabile solo nelle settimane in cui era davvero "
                        f"nell'indice, delistati inclusi. {skipped} segnali scartati per "
                        f"questo motivo. <b>Il survivorship bias è eliminato.</b></p>")
        elif "index_static" in modes.values():
            uni_note = (f"<p class='warn'>⚠ <b>Universo statico</b> ({n} titoli, composizione "
                        f"ATTUALE dell'indice applicata a tutto il periodo). Stai testando "
                        f"aziende che sappiamo essere sopravvissute: i risultati vanno letti "
                        f"come <b>limite superiore</b> della performance reale "
                        f"(survivorship bias).</p>")
    doc = f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weinstein Bot — report di backtest</title>
<style>
  :root {{ --ink:{INK}; --paper:{PAPER}; --grid:{GRID}; --strat:{STRAT}; --bench:{BENCH}; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink);
         font:15px/1.55 Georgia,'Times New Roman',serif; }}
  main {{ max-width:1060px; margin:0 auto; padding:2.2rem 1.4rem 4rem; }}
  header {{ border-bottom:3px double var(--ink); padding-bottom:1rem; margin-bottom:1.6rem; }}
  h1 {{ font-size:1.9rem; margin:0 0 .2rem; font-weight:600; letter-spacing:.01em; }}
  .meta {{ font:12px/1.4 ui-monospace,Menlo,Consolas,monospace; color:#5d5647; }}
  h2 {{ font-size:1.15rem; margin:2.4rem 0 .7rem; border-bottom:1px solid var(--grid);
        padding-bottom:.25rem; }}
  table {{ border-collapse:collapse; width:100%; font-size:.92rem; }}
  th,td {{ padding:.4rem .6rem; border-bottom:1px solid var(--grid); text-align:left; }}
  th {{ font:600 .78rem ui-monospace,Menlo,Consolas,monospace; text-transform:uppercase;
        letter-spacing:.06em; color:#5d5647; }}
  td.num {{ font-family:ui-monospace,Menlo,Consolas,monospace; text-align:right; }}
  .chip {{ display:inline-block; border:1px solid; border-radius:3px; padding:.05rem .45rem;
           font:12px ui-monospace,monospace; }}
  .note {{ font-size:.82rem; color:#5d5647; }}
  .warn {{ background:#f7ecd8; border-left:4px solid #e0a458; padding:.6rem .9rem; }}
  .ok {{ background:#e8f0e4; border-left:4px solid #7fb069; padding:.6rem .9rem; }}
  .tablewrap {{ max-height:420px; overflow:auto; border:1px solid var(--grid); }}
  textarea {{ width:100%; min-height:340px; font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;
              background:#fff; color:var(--ink); border:1px solid var(--grid); padding:.8rem; }}
  button {{ font:600 .85rem Georgia,serif; background:var(--ink); color:var(--paper);
            border:0; padding:.55rem 1.1rem; cursor:pointer; border-radius:3px; }}
  button:hover {{ background:var(--strat); }}
  footer {{ margin-top:3rem; border-top:1px solid var(--grid); padding-top:1rem;
            font-size:.8rem; color:#5d5647; }}
</style></head><body><main>

<header>
  <h1>Weinstein Bot — report di backtest</h1>
  <div class="meta">generato {generated} · provider: {provider} ·
  periodo {cfg['backtest']['start']} → {cfg['backtest']['end']} ·
  split OOS {cfg['backtest']['oos_split']} · segnali: {m['n_signals']} ·
  operazioni: {m['n_trades']}</div>
</header>

{caveat_synth}
{uni_note}

<h2>1 · Risultato di portafoglio</h2>
{_metrics_table(m)}
<p class="note">Operazioni chiuse: {m['n_trades']} · hit rate: {_fmt_pct(m['hit_rate_trades'])}
 · guadagno medio: {_fmt_pct(m['avg_win'])} · perdita media: {_fmt_pct(m['avg_loss'])}.
 Costi inclusi: {cfg['portfolio']['commission_bps']} bps commissioni +
 {cfg['portfolio']['slippage_bps']} bps slippage per lato.</p>
{plots[0]}

<h2>2 · Qualità dei segnali (indipendente dal sizing)</h2>
{_signal_quality_table(m, horizons)}

<h2>3 · Titolo d'esempio: la banda delle fasi</h2>
<p>{stage_legend}</p>
{plots[1]}
{plots[2]}

<h2>4 · Registro dei segnali</h2>
<div class="tablewrap">{_signals_table(res.signal_stats, horizons)}</div>

<h2>5 · Configurazione di questo run</h2>
<p class="note">Il report è autodocumentante: qui sotto c'è la config esatta usata.
Modificala e scaricala, poi rilancia: <code>python run.py backtest --config nuova.yaml</code></p>
<textarea id="cfgbox" spellcheck="false">{cfg_yaml}</textarea>
<p><button onclick="exportCfg()">Scarica config modificata</button></p>

<footer>
Metodo: Stan Weinstein, <i>Secrets for Profiting in Bull and Bear Markets</i> (1988),
implementazione parametrica con filtro di mercato e RS Mansfield obbligatoria.
Limiti da ricordare: risultati storici ≠ risultati futuri; se il dataset esclude i titoli
delistati, i numeri vanno letti come limite superiore (survivorship bias);
gli stop sono valutati sulla chiusura settimanale, coerentemente col metodo.
Questo strumento produce candidati di screening, non consigli di investimento.
</footer>

<script>
function exportCfg() {{
  const blob = new Blob([document.getElementById('cfgbox').value],
                        {{type:'text/yaml'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'config_modificata.yaml';
  a.click();
  URL.revokeObjectURL(a.href);
}}
</script>
</main></body></html>"""

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return out_path
