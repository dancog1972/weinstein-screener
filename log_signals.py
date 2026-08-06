#!/usr/bin/env python3
"""Follow-up dei segnali: PIENI automatici (dallo screener) + QUASI seguiti a mano
(watchlist Cloudflare). Per ogni segnale simula l'USCITA col metodo VALIDATO del
backtest — stop iniziale → trailing MA30-ATR → rottura MA30 — e mostra uscita,
motivo, risultato, più dove sta il titolo ORA. Grafici cliccabili con entry/stop/
uscita. Pagina: output/signals.html. signals_log.json accumula solo i pieni.

Non chiama l'API: legge config + cache locale (+ il Worker per la watchlist).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

from src.console import setup as _console_setup

_console_setup()

import numpy as np
import pandas as pd
import requests
import yaml

from src.config import load_config
from src.data_store import DataStore
from src.indicators import enrich_weekly, to_weekly
from src.providers import make_provider
from src.signals import prepare_ticker
from make_charts import STAGE_COLORS


def _fetch_watchlist() -> list[dict]:
    """I 'quasi' seguiti a mano, dal Worker Cloudflare (WATCH_URL + WATCH_SECRET)."""
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


def _finite(a) -> list:
    return [None if not np.isfinite(x) else round(float(x), 4) for x in a]


def _series(win: pd.DataFrame) -> dict:
    bl = win["base_len"].to_numpy()
    res = win["resistance"].to_numpy()
    return {
        "t": [t.strftime("%Y-%m-%d") for t in win.index],
        "o": _finite(win["adj_open"].to_numpy()), "h": _finite(win["adj_high"].to_numpy()),
        "l": _finite(win["adj_low"].to_numpy()), "c": _finite(win["adj_close"].to_numpy()),
        "ma": _finite(win["ma"].to_numpy()), "vol": _finite(win["volume"].to_numpy()),
        "stage": [int(x) if np.isfinite(x) else 0 for x in win["stage"].to_numpy()],
        "res": [None if (b <= 0 or not np.isfinite(r)) else round(float(r), 4)
                for r, b in zip(res, bl)],
    }


def _simulate_exit(df: pd.DataFrame, sig: pd.Timestamp, entry_ref: float,
                   stop_ref: float, x: dict) -> dict:
    """Replica l'uscita del backtest (long): ogni settimana DOPO il segnale alza lo
    stop col trailing MA30-ATR (solo in su), ed esce se la chiusura scende sotto lo
    stop ('stop'/'trailing stop') o sotto la MA30 con pendenza negativa ('MA30
    breakdown'). Se ancora aperto, ritorna cur_stop = stop trailing corrente.
    Tutto su scala di aggiustamento coerente (entry_ref/stop_ref dal df attuale)."""
    idx = df.index
    start = int(idx.searchsorted(sig, side="right"))     # prima settimana dopo il segnale
    cur_stop, raised = float(stop_ref), False
    mode = x.get("trailing_mode")
    k_atr = x.get("ma_trail_atr_mult")
    buf = x.get("ma_trail_buffer", 0.05)
    intraweek = x.get("stop_intraweek", True)
    ma_bd = x.get("ma_breakdown", True)
    for i in range(start, len(idx)):
        row = df.iloc[i]
        close = float(row["adj_close"])
        ma = row.get("ma"); atr = row.get("atr"); slope = row.get("ma_slope")
        if mode == "ma30" and pd.notna(ma):
            dist = (k_atr * float(atr)) if (k_atr is not None and pd.notna(atr) and np.isfinite(atr)) \
                else float(ma) * buf
            ns = float(ma) - dist
            if ns > cur_stop:
                cur_stop, raised = ns, True
        low = float(row["adj_low"]) if pd.notna(row.get("adj_low")) else close
        touched = (low <= cur_stop) if intraweek else (close <= cur_stop)
        if touched:
            px = cur_stop
            op = float(row["adj_open"]) if pd.notna(row.get("adj_open")) else close
            if op < cur_stop:
                px = op
            return {"status": "exited", "exit_date": idx[i].strftime("%Y-%m-%d"),
                    "exit_price": round(px, 2), "exit_reason": "trailing stop" if raised else "stop",
                    "realized_pct": round((px / entry_ref - 1) * 100, 1), "cur_stop": round(cur_stop, 2)}
        if ma_bd and pd.notna(ma) and close < float(ma) and pd.notna(slope) and slope < 0:
            return {"status": "exited", "exit_date": idx[i].strftime("%Y-%m-%d"),
                    "exit_price": round(close, 2), "exit_reason": "MA30 breakdown",
                    "realized_pct": round((close / entry_ref - 1) * 100, 1), "cur_stop": round(cur_stop, 2)}
    return {"status": "open", "cur_stop": round(cur_stop, 2)}


def _analyze(store: DataStore, cfg: dict, bench_cache: dict, markets: dict,
             e: dict, want_series: bool) -> dict | None:
    """Prezzo attuale, % dal segnale, e simulazione dell'USCITA per un segnale.
    Calcola il settimanale come lo screener (prepare_ticker con benchmark)."""
    st = cfg["stages"]
    today = pd.Timestamp(date.today())
    start = (today - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    bsym = (markets.get(e.get("market")) or {}).get("benchmark", "SPY.US")
    if bsym not in bench_cache:
        try:
            bench_cache[bsym] = enrich_weekly(to_weekly(store.get_with_warmup(bsym, start, end)),
                                              None, st["ma_weeks"], st["slope_lookback"])
        except Exception:  # noqa: BLE001
            bench_cache[bsym] = None
    try:
        d = store.get_with_warmup(e["ticker"], start, end)
        df = prepare_ticker(enrich_weekly(to_weekly(d), bench_cache[bsym],
                                          st["ma_weeks"], st["slope_lookback"]), st["flat_slope"])
    except Exception:  # noqa: BLE001
        return None
    if df.empty:
        return None
    ac = df["adj_close"].dropna()
    sig = pd.Timestamp(e["signal_date"])
    at = ac.asof(sig)
    if at is None or not (at > 0):
        return None
    at = float(at)
    cur = float(ac.iloc[-1])
    out = {"last_price": round(cur, 2), "pct_since": round((cur / at - 1) * 100, 1),
           "last_data": ac.index[-1].strftime("%Y-%m-%d")}
    entry, stop = float(e.get("entry", 0)), float(e.get("stop", 0))
    if entry > 0 and stop > 0:
        stop_ref = at * (stop / entry)                    # rischio recorded, su scala attuale
        out.update(_simulate_exit(df, sig, at, stop_ref, cfg["exits"]))
    if want_series:
        win = df.iloc[-208:] if len(df) > 208 else df
        out["series"] = _series(win)
        out["entry_ref"] = round(at, 2)                   # entry su scala del grafico
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Follow-up dei segnali: pieni + quasi seguiti, con uscite")
    ap.add_argument("--config", default="config_us_liquidity.yaml")
    ap.add_argument("--markets", default="config_screener_markets.yaml")
    ap.add_argument("--screener", default="output/screener.json")
    ap.add_argument("--log", default="signals_log.json")
    ap.add_argument("--out", default="output/signals.html")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.markets, encoding="utf-8") as fh:
        markets = yaml.safe_load(fh)["markets"]
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    bench_cache: dict = {}
    today = date.today().isoformat()

    logp = Path(args.log)
    log: list[dict] = json.loads(logp.read_text(encoding="utf-8")) if logp.exists() else []
    seen = {(e["ticker"], e["signal_date"]) for e in log}

    data = json.loads(Path(args.screener).read_text(encoding="utf-8"))
    added = 0
    for c in data.get("candidates", []):
        if c.get("kind") != "pieno" or (c["ticker"], c["date"]) in seen:
            continue
        log.append({"ticker": c["ticker"], "market": c["market"], "signal_date": c["date"],
                    "entry": c["entry"], "stop": c["stop_bot"], "base_len": c["base_len"],
                    "mansfield": c["mansfield"], "vol_ratio": c["vol_ratio"],
                    "currency": c.get("currency", ""), "first_logged": today})
        seen.add((c["ticker"], c["date"]))
        added += 1

    for e in log:
        a = _analyze(store, cfg, bench_cache, markets, e, want_series=True)
        if a:
            e.update(a)
        e["weeks_since"] = (pd.Timestamp(today) - pd.Timestamp(e["signal_date"])).days // 7
        e["last_updated"] = today

    # il log committato NON contiene le serie (pesanti): solo i campi leggeri
    slim = [{k: v for k, v in e.items() if k != "series"} for e in log]
    logp.write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")

    quasi = _fetch_watchlist()
    for w in quasi:
        a = _analyze(store, cfg, bench_cache, markets, w, want_series=True)
        if a:
            w.update(a)
        sd = w.get("signal_date")
        w["weeks_since"] = (pd.Timestamp(today) - pd.Timestamp(sd)).days // 7 if sd else None

    log.sort(key=lambda e: e["signal_date"], reverse=True)
    quasi.sort(key=lambda w: w.get("signal_date") or "", reverse=True)
    _write_html(Path(args.out), log, quasi, today)
    print(f"✓ Follow-up: {len(log)} pieni ({added} nuovi) · {len(quasi)} quasi → {args.out}")


def _write_html(path: Path, pieni: list[dict], quasi: list[dict], today: str) -> None:
    watch_url = os.environ.get("WATCH_URL", "")

    def light(e: dict) -> dict:
        keep = ("ticker", "market", "signal_date", "entry", "stop", "mansfield", "currency",
                "last_price", "pct_since", "status", "cur_stop", "exit_date", "exit_price",
                "exit_reason", "realized_pct", "series")
        return {k: e.get(k) for k in keep}

    payload = {
        "updated": today, "watch_url": watch_url,
        "stageColors": {str(k): v[0] for k, v in STAGE_COLORS.items()},
        "pieni": [light(e) for e in pieni],
        "quasi": {f"{w.get('ticker')}|{w.get('signal_date')}": light(w) for w in quasi},
    }
    html = _HTML.replace("/*DATA*/", json.dumps(payload, ensure_ascii=False))
    plotly = Path("assets/plotly.min.js")
    if plotly.exists():
        html = html.replace(
            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>',
            f"<script>{plotly.read_text(encoding='utf-8')}</script>")
    path.write_text(html, encoding="utf-8")


_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signals follow-up</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
 body{background:#151b21;color:#d7e0e6;font:13px/1.5 -apple-system,Segoe UI,sans-serif;margin:0;padding:24px}
 h1{font-size:19px;margin:0 0 4px} .sub{color:#7f8c98;font-size:12px;margin-bottom:14px}
 h2{font-size:14px;margin:22px 0 4px} h2 small{color:#7f8c98;font-weight:400;font-size:11px}
 .cnt{background:#233240;border-radius:9px;padding:0 7px;font-size:11px;color:#d7e0e6}
 a{color:#6fe3a1;text-decoration:none} a:hover{text-decoration:underline}
 table{border-collapse:collapse;width:100%;font-size:12px;margin-top:6px}
 th{color:#7f8c98;font-weight:600;border-bottom:1px solid #2b353f;padding:7px 8px;white-space:nowrap;text-align:left}
 th.n{text-align:right}
 td{border-bottom:1px solid #222c35;padding:7px 8px} tr.row{cursor:pointer} tr.row:hover td{background:#1a2128}
 tr.row.sel td{background:#1d2731}
 .n{text-align:right;font-family:SF Mono,Consolas,monospace} .tk{font-weight:600;color:#6fe3a1}
 .stop{color:#d6604d} .pos{color:#6fe3a1} .neg{color:#d6604d} .op{color:#8fb8d8} .ex{color:#e0a458}
 .none{text-align:center;color:#7f8c98;padding:20px} i{color:#7f8c98}
 .mk{background:#233240;color:#8fb8d8;border-radius:3px;padding:1px 6px;font-size:10px;font-weight:700}
 #chart{width:100%;height:430px;margin-top:8px}
 .card{background:#1a2128;border:1px solid #2b353f;border-radius:8px;padding:12px;margin:10px 0}
 .chd{font-weight:600;color:#6fe3a1;margin-bottom:2px} .chd small{color:#7f8c98;font-weight:400}
 .foot{color:#7f8c98;font-size:11px;margin-top:18px;line-height:1.7}
</style></head><body>
<h1>Signals follow-up</h1>
<div class="sub" id="sub"></div>
<h2>FULL signals <span class="cnt" id="pcnt">0</span> <small>automatic, from the screener</small></h2>
<div id="pieni"></div>
<h2>Followed NEAR <span class="cnt" id="qcnt">0</span> <small>your picks — live from the watchlist (★ in the screener)</small></h2>
<div id="quasi"></div>
<div id="detail"></div>
<div class="foot">
Each signal is followed with the <b>validated exit</b>: initial pivot stop → trailing MA30-ATR (rises with the
trend) → MA30 breakdown, whichever hits first. <b>Stop now</b> = current trailing stop (where you'd exit if still
open). <b>Result</b> = realized return at exit. <b>% since</b> keeps running (where the stock is now vs the signal
— useful to see a recovery after the stop, or a late entry). Not a portfolio: nothing is bought here.
Click a row to see the chart with entry, stop and exit.
</div>
<script>
const DATA = /*DATA*/, SC = DATA.stageColors, WATCH_URL = DATA.watch_url;
const $ = id => document.getElementById(id);
let sel = null;
$("sub").innerHTML = `Updated ${DATA.updated} · <a href="metodo.html">the method</a> · <a href="index.html">← back to screener</a>`;

function n2(v,d){ return (typeof v==='number')?(+v).toFixed(d==null?2:d):'–'; }
function pctc(v){ if(v==null) return "<td class='n'>–</td>"; const c=v>0?'pos':v<0?'neg':''; return `<td class='n ${c}'>${v>0?'+':''}${(+v).toFixed(1)}%</td>`; }
const HEAD = "<tr><th>Ticker</th><th>Mkt</th><th>Signal</th><th class='n'>Entry</th><th class='n'>Stop now</th>"
  +"<th>Exit</th><th>Reason</th><th class='n'>Result</th><th class='n'>Now</th><th class='n'>% since</th><th class='n'>Mans</th></tr>";

function rowHtml(d, key){
  const exited = d.status==='exited';
  const exitCell = exited ? `<td class='ex'>${d.exit_date||''}</td><td class='ex'>${d.exit_reason||''}</td>`
                          : `<td class='op'>open</td><td class='op'>—</td>`;
  const result = exited ? pctc(d.realized_pct) : "<td class='n'>–</td>";
  const pending = (d.last_price==null);
  return `<tr class="row ${sel===key?'sel':''}" data-k="${key}">`
    +`<td class='tk'>${d.ticker}</td><td><span class='mk'>${d.market||''}</span></td>`
    +`<td>${d.signal_date}</td><td class='n'>${n2(d.entry)}</td>`
    +`<td class='n stop'>${n2(d.cur_stop)}</td>${exitCell}${result}`
    +`<td class='n'>${pending?"<i>pending</i>":n2(d.last_price)}</td>${pctc(d.pct_since)}`
    +`<td class='n'>${n2(d.mansfield,1)}</td></tr>`;
}
function tableHtml(items, empty){
  const body = items.map(([k,d])=>rowHtml(d,k)).join("") || `<tr><td colspan='11' class='none'>${empty}</td></tr>`;
  return `<table>${HEAD}<tbody>${body}</tbody></table>`;
}
function bind(container){
  container.querySelectorAll("tr.row").forEach(tr=>tr.onclick=()=>{ sel=tr.dataset.k; render(); });
}
function findData(key){
  const p = DATA.pieni.find(d=>d.ticker+"|"+d.signal_date===key);
  return p || DATA.quasi[key] || (LIVE[key]||null);
}
let LIVE = {};   // quasi live dal Worker (merge con DATA.quasi che ha serie+uscita)

function renderPieni(){
  const items = DATA.pieni.map(d=>[d.ticker+"|"+d.signal_date, d]);
  $("pcnt").textContent = items.length;
  $("pieni").innerHTML = tableHtml(items, "No full signals recorded yet.");
  bind($("pieni"));
}
function renderQuasi(){
  const items = Object.entries(LIVE);
  $("qcnt").textContent = items.length;
  $("quasi").innerHTML = tableHtml(items, "No followed near-misses. In the screener open a candidate and click ★ follow.");
  bind($("quasi"));
}
function renderDetail(){
  const d = sel ? findData(sel) : null;
  if(!d || !d.series){ $("detail").innerHTML = sel ? `<div class="card"><div class="none">chart pending — will appear at the next update</div></div>` : ""; return; }
  $("detail").innerHTML = `<div class="card"><div class="chd">${d.ticker} <small>· signal ${d.signal_date} · ${d.status==='exited'?('exited '+d.exit_date+' ('+d.exit_reason+')'):'open'}</small></div><div id="chart"></div></div>`;
  drawChart(d);
}
function render(){ renderPieni(); renderQuasi(); renderDetail(); }

function drawChart(d){
  const s=d.series, entry=d.entry_ref!=null?d.entry_ref:d.entry;
  const traces=[
    {type:"candlestick",x:s.t,open:s.o,high:s.h,low:s.l,close:s.c,name:"",increasing:{line:{color:"#2ca25f"}},decreasing:{line:{color:"#d6604d"}},xaxis:"x",yaxis:"y"},
    {type:"scatter",x:s.t,y:s.ma,mode:"lines",line:{color:"#2166ac",width:1.5},name:"MA30",xaxis:"x",yaxis:"y"},
    {type:"scatter",x:s.t,y:s.res,mode:"lines",line:{color:"#8c510a",width:1.2,dash:"dash"},name:"resistance",xaxis:"x",yaxis:"y"},
    {type:"bar",x:s.t,y:s.vol,marker:{color:s.c.map((c2,i)=>c2>=s.o[i]?"#2ca25f":"#d6604d"),opacity:.6},name:"volume",xaxis:"x",yaxis:"y2"},
  ];
  const shapes=[]; let i0=0;
  for(let i=1;i<=s.stage.length;i++){ if(i===s.stage.length||s.stage[i]!==s.stage[i0]){ const col=SC[s.stage[i0]];
    if(col&&s.stage[i0]!==0) shapes.push({type:"rect",xref:"x",yref:"paper",x0:s.t[i0],x1:s.t[Math.min(i,s.t.length-1)],y0:0,y1:1,fillcolor:col,opacity:.28,line:{width:0},layer:"below"}); i0=i; } }
  const end=s.t[s.t.length-1];
  // current/exit stop line
  if(d.cur_stop!=null) shapes.push({type:"line",xref:"x",yref:"y",x0:d.signal_date,x1:d.status==='exited'?d.exit_date:end,y0:d.cur_stop,y1:d.cur_stop,line:{color:"#b2182b",width:1.3,dash:"dot"}});
  const ann=[{x:d.signal_date,y:entry,xref:"x",yref:"y",text:"ENTRY "+n2(entry),showarrow:true,arrowcolor:"#0c8f4d",arrowhead:6,ax:0,ay:-26,font:{color:"#0c8f4d",size:11}}];
  if(d.status==='exited'){
    ann.push({x:d.exit_date,y:d.exit_price,xref:"x",yref:"y",text:"EXIT "+n2(d.exit_price)+" · "+d.exit_reason,showarrow:true,arrowcolor:"#e0a458",arrowhead:6,ax:0,ay:-26,font:{color:"#e0a458",size:11}});
    shapes.push({type:"line",xref:"x",yref:"y",x0:d.exit_date,x1:d.exit_date,y0:0,y1:1,yref2:"paper",line:{color:"#e0a458",width:1,dash:"dot"}});
  } else {
    ann.push({x:end,y:d.cur_stop,xref:"x",yref:"y",text:"stop "+n2(d.cur_stop),showarrow:false,xanchor:"right",yanchor:"top",font:{color:"#d6604d",size:10}});
  }
  const layout={paper_bgcolor:"#1a2128",plot_bgcolor:"#1a2128",font:{color:"#d7e0e6",size:11},margin:{l:48,r:16,t:10,b:26},showlegend:true,legend:{orientation:"h",y:1.02,x:0,font:{size:10}},
    xaxis:{domain:[0,1],rangeslider:{visible:false},gridcolor:"#222c35",anchor:"y2"},
    yaxis:{domain:[0.26,1],gridcolor:"#222c35",title:{text:"price"}},yaxis2:{domain:[0,0.2],gridcolor:"#222c35",title:{text:"vol"}},
    shapes,annotations:ann,dragmode:"zoom"};
  Plotly.newPlot("chart",traces,layout,{responsive:true,scrollZoom:true,modeBarButtonsToRemove:["select2d","lasso2d"]});
}

async function loadWatch(){
  // parte dai quasi analizzati dal job (con serie+uscita); poi il Worker aggiunge/toglie i live
  LIVE = {...DATA.quasi};
  if(WATCH_URL){ const sec=localStorage.getItem("watch_secret");
    if(sec){ try{
      const r=await fetch(WATCH_URL.replace(/\/$/,"")+"/list?secret="+encodeURIComponent(sec));
      if(r.ok){ const list=await r.json(); const live={};
        list.forEach(w=>{ const k=w.ticker+"|"+w.signal_date; live[k]=DATA.quasi[k]||{ticker:w.ticker,market:w.market,signal_date:w.signal_date,entry:w.entry,stop:w.stop,mansfield:w.mansfield,currency:w.currency,status:"open"}; });
        LIVE=live;
      }
    }catch(e){} }
  }
  render();
}
loadWatch();
</script></body></html>
"""


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
