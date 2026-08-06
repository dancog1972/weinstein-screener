#!/usr/bin/env python3
"""Costruisce la PAGINA INTERATTIVA dello screener a partire da un run già fatto.

Legge `output/screener.json` (candidati + metriche prodotti da screener.py) e per
ogni candidato estrae dalla cache le serie SETTIMANALI (208 sett. = 4 anni) per i
grafici Plotly. Produce un HTML autonomo con:
  - tab per mercato
  - lista candidati per tab (ordinabile)
  - click su un candidato -> scheda con candlestick Plotly ZOOMABILE
  - pannello SLIDER delle soglie: ricalcola pieno/quasi e motivi di scarto DAL VIVO,
    in browser (nessun backend). È il "modo custom".

Uso (offline, dalla cache):
    $env:EODHD_API_KEY = "cache-only"
    python build_page.py --asof 2026-08-05

Destinato a GitHub Pages: Plotly da CDN, resto embedded.
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
import yaml

from src.config import load_config
from src.data_store import DataStore
from src.indicators import enrich_weekly, to_weekly
from src.providers import make_provider
from src.signals import prepare_ticker
from make_charts import STAGE_COLORS

STAGE_NAME = {0: "—", 1: "Base", 2: "Avanzata", 3: "Top", 4: "Declino"}

# filtri della config che NON sono controllati da slider/toggle nella pagina:
# li teniamo com'erano al momento del run (non ricalcolabili dai soli scalari).
_FIXED_FAILS = {"Mansfield not rising", "news-driven breakout",
                "base volume not contracted", "base no dry-up/accumulation"}


def _finite_list(a: np.ndarray) -> list:
    return [None if not np.isfinite(x) else round(float(x), 4) for x in a]


def _series(win: pd.DataFrame) -> dict:
    bl = win["base_len"].to_numpy()
    res = win["resistance"].to_numpy()
    return {
        "t": [ts.strftime("%Y-%m-%d") for ts in win.index],
        "o": _finite_list(win["adj_open"].to_numpy()),
        "h": _finite_list(win["adj_high"].to_numpy()),
        "l": _finite_list(win["adj_low"].to_numpy()),
        "c": _finite_list(win["adj_close"].to_numpy()),
        "ma": _finite_list(win["ma"].to_numpy()),
        "vol": _finite_list(win["volume"].to_numpy()),
        "stage": [int(x) if np.isfinite(x) else 0 for x in win["stage"].to_numpy()],
        # la resistenza si disegna solo dove esiste una base
        "res": [None if (b <= 0 or not np.isfinite(r)) else round(float(r), 4)
                for r, b in zip(res, bl)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Costruisce la pagina interattiva dello screener")
    ap.add_argument("--config", default="config_us_liquidity.yaml")
    ap.add_argument("--markets", default="config_screener_markets.yaml")
    ap.add_argument("--json", default="output/screener_app_data.json",
                    help="superset prodotto dallo screener (righe filtrabili dagli slider)")
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--asof", default=None, help="data di riferimento (default oggi)")
    ap.add_argument("--series-cap", type=int, default=150,
                    help="per quanti candidati (i migliori) caricare il grafico Plotly; "
                         "gli altri restano righe filtrabili senza grafico")
    ap.add_argument("--out", default="output/screener_app.html")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.markets, encoding="utf-8") as fh:
        markets = yaml.safe_load(fh)["markets"]
    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    meta, cands = data["meta"], data["candidates"]

    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    st = cfg["stages"]
    today = pd.Timestamp(args.asof) if args.asof else pd.Timestamp(date.today())
    start = (today - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    bench_wk: dict = {}

    def get_bench(sym: str):
        if sym not in bench_wk:
            d = store.get_with_warmup(sym, start, end)
            bench_wk[sym] = enrich_weekly(to_weekly(d), None, st["ma_weeks"], st["slope_lookback"])
        return bench_wk[sym]

    # Carico il GRAFICO solo per `series-cap` candidati (per non imbarcare centinaia
    # di serie). PRIORITÀ ai più "vicini": meno filtri falliti prima, poi per forza
    # relativa → i candidati della vista di default (fail ≤1) hanno SEMPRE il grafico;
    # restano senza solo i quasi profondi del superset (slider a 3-4).
    cap = args.series_cap
    prio = sorted(range(len(cands)),
                  key=lambda i: (len(cands[i].get("fails", [])), -cands[i].get("mansfield", 0)))
    charted = set(prio[:cap])
    print(f"Estraggo le serie per {min(cap, len(cands))}/{len(cands)} candidati (i più vicini)...")
    for i, c in enumerate(cands):
        if i in charted:
            try:
                bench = get_bench(markets[c["market"]]["benchmark"])
                d = store.get_with_warmup(c["ticker"], start, end)
                wk = prepare_ticker(
                    enrich_weekly(to_weekly(d), bench, st["ma_weeks"], st["slope_lookback"]),
                    st["flat_slope"])
                win = wk.iloc[-208:] if len(wk) > 208 else wk
                c["series"] = _series(win)
            except Exception as e:  # noqa: BLE001
                print(f"  [!] {c['ticker']}: serie non disponibile ({e})")
                c["series"] = None
        else:
            c["series"] = None
        # flag ricalcolabili lato client
        f = set(c.get("fails", []))
        c["f_market"] = "market not in Stage 2" in f
        c["f_sector"] = "sector not in Stage 2" in f
        c["f_decline"] = "base not after decline" in f
        c["fixed_fails"] = sorted(f & _FIXED_FAILS)

    # raggruppo per mercato, nell'ordine del file mercati
    by_market = []
    for name in markets:
        info = next((m for m in meta["markets"] if m["name"] == name), None)
        if info is None:
            continue
        rows = [c for c in cands if c["market"] == name]
        by_market.append({**info, "candidates": rows})

    sig = cfg["signal"]
    defaults = {
        "baseMin": sig["min_base_weeks"], "depthMax": round(sig["max_base_depth"] * 100),
        "volMin": sig["volume_ratio_min"], "mansMin": sig["mansfield_min"],
        "riskMax": 60, "reqMarket": bool(sig.get("market_filter", True)),
        "reqSector": bool(sig.get("sector_filter", True) and meta.get("sector_active")),
        "reqDecline": bool(sig.get("require_decline", True)),
    }
    payload = {"meta": meta, "markets": by_market, "defaults": defaults,
               "watch_url": os.environ.get("WATCH_URL", ""),   # Worker Cloudflare (vuoto = niente ★)
               "stageColors": {str(k): v[0] for k, v in STAGE_COLORS.items()},
               "stageNames": {str(k): v for k, v in STAGE_NAME.items()}}

    html = _HTML.replace("/*DATA*/", json.dumps(payload, ensure_ascii=False))
    # Plotly: se assets/plotly.min.js c'è lo INGLOBIAMO (pagina autonoma, niente
    # CDN → funziona offline, nessuna dipendenza esterna anche su Pages). Altrimenti
    # si ripiega sul tag CDN già presente.
    plotly = Path("assets/plotly.min.js")
    if plotly.exists():
        js = plotly.read_text(encoding="utf-8")
        html = html.replace(
            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>',
            f'<script>{js}</script>')
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(html, encoding="utf-8")
    print(f"✓ Pagina: {outp}")


_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weinstein Screener — interactive</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
 :root{--bg:#151b21;--card:#1a2128;--line:#2b353f;--fg:#d7e0e6;--mut:#7f8c98;
       --grn:#6fe3a1;--grnbg:#16341f;--amb:#e0a458;--ambbg:#3a2a16;--red:#d6604d;--mk:#8fb8d8}
 *{box-sizing:border-box}
 body{background:var(--bg);color:var(--fg);font:13px/1.5 -apple-system,Segoe UI,sans-serif;margin:0;padding:20px}
 h1{font-size:19px;margin:0 0 4px} .sub{color:var(--mut);font-size:12px;margin-bottom:14px}
 a{color:var(--grn);text-decoration:none} a:hover{text-decoration:underline}
 /* pannello soglie */
 .panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin-bottom:16px}
 .panel h2{font-size:12px;text-transform:uppercase;color:var(--mut);margin:0 0 10px;letter-spacing:.04em}
 .ctrls{display:flex;flex-wrap:wrap;gap:18px;align-items:flex-end}
 .ctrl{display:flex;flex-direction:column;gap:3px;min-width:150px}
 .ctrl label{font-size:11px;color:var(--mut)} .ctrl b{color:var(--fg)}
 .ctrl input[type=range]{width:150px;accent-color:var(--grn)}
 .togs{display:flex;gap:14px;align-items:center} .tog{display:flex;gap:5px;align-items:center;font-size:12px}
 .tog input{accent-color:var(--grn)} .reset{margin-left:auto;background:#233240;color:var(--fg);
   border:1px solid var(--line);border-radius:5px;padding:6px 12px;cursor:pointer;font-size:12px}
 .reset:hover{border-color:var(--mut)}
 /* tab */
 .tabs{display:flex;gap:4px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin-bottom:12px}
 .tab{background:none;border:none;color:var(--mut);padding:9px 14px;cursor:pointer;font-size:13px;
      border-bottom:2px solid transparent;font-weight:600}
 .tab.on{color:var(--grn);border-bottom-color:var(--grn)}
 .tab .cnt{background:#233240;border-radius:9px;padding:0 7px;margin-left:5px;font-size:11px;color:var(--fg)}
 .mktbar{font-size:12px;color:var(--mut);margin:0 0 10px} .mktbar b{color:var(--fg)}
 /* lista */
 table{border-collapse:collapse;width:100%;font-size:12px} th{text-align:left;color:var(--mut);font-weight:600;
   border-bottom:1px solid var(--line);padding:7px 8px;white-space:nowrap;cursor:pointer;user-select:none}
 th:hover{color:var(--fg)} td{border-bottom:1px solid #222c35;padding:7px 8px}
 tr.row{cursor:pointer} tr.row:hover td{background:#1a2128} tr.row.sel td{background:#1d2731}
 .n{text-align:right;font-family:SF Mono,Consolas,monospace} .tk{font-weight:600;color:var(--grn)}
 .stop{color:var(--red)}
 .bp{background:var(--grnbg);color:var(--grn);border:1px solid #2f6b45;border-radius:3px;padding:1px 6px;font-size:10px;font-weight:700}
 .bq{background:var(--ambbg);color:var(--amb);border:1px solid #6b5330;border-radius:3px;padding:1px 6px;font-size:10px;font-weight:600}
 .fl{color:var(--amb);font-size:11px} .none{color:var(--mut);text-align:center;padding:22px}
 /* scheda */
 .card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin-top:14px}
 .chd{display:flex;align-items:baseline;gap:10px;margin-bottom:6px} .chd .tk{font-size:16px}
 .follow{margin-left:auto;background:#233240;color:#8fb8d8;border:1px solid #2b353f;border-radius:5px;padding:3px 12px;cursor:pointer;font-size:12px;align-self:center}
 .follow:hover{border-color:#6fe3a1} .follow.on{background:#16341f;color:#6fe3a1;border-color:#2f6b45}
 .chd .meta{color:var(--mut);font-size:11px}
 #chart{width:100%;height:440px}
 .kv{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
 .kv div{background:var(--bg);border-radius:4px;padding:5px 10px;min-width:74px;border:1px solid transparent}
 .kv div.pass{border-color:#2f6b45} .kv div.fail{border-color:#7a3b34}
 .kv b{display:block;color:var(--mut);font-size:10px;font-weight:600;text-transform:uppercase}
 .kv span{font-family:SF Mono,Consolas,monospace;font-size:13px}
 .kv .ok{color:var(--grn)} .kv .no{color:var(--red)}
 .hint{color:var(--mut);font-size:11px;margin-top:6px}
</style></head><body>
<h1>Weinstein Screener — <span style="color:var(--grn)">interactive</span></h1>
<div class="sub" id="hdr"></div>
<div class="sub"><a href="signals.html">Signals follow-up &rarr;</a> &nbsp;·&nbsp; <a href="metodo.html">the method</a></div>

<div class="panel">
 <h2>Custom thresholds — recompute full/near live</h2>
 <div class="ctrls">
  <div class="ctrl"><label>base ≥ <b id="lBase"></b> w</label><input id="sBase" type="range" min="4" max="30" step="1"></div>
  <div class="ctrl"><label>depth ≤ <b id="lDepth"></b>%</label><input id="sDepth" type="range" min="10" max="45" step="1"></div>
  <div class="ctrl"><label>volume ≥ <b id="lVol"></b>×</label><input id="sVol" type="range" min="1" max="4" step="0.1"></div>
  <div class="ctrl"><label>Mansfield ≥ <b id="lMans"></b></label><input id="sMans" type="range" min="-5" max="20" step="0.5"></div>
  <div class="ctrl"><label>risk ≤ <b id="lRisk"></b>%</label><input id="sRisk" type="range" min="10" max="60" step="1"></div>
  <div class="ctrl"><label>show up to <b id="lMaxF"></b> failed filters</label><input id="sMaxF" type="range" min="0" max="4" step="1"></div>
  <div class="togs">
   <label class="tog"><input id="tMarket" type="checkbox"> market Stage 2</label>
   <label class="tog"><input id="tSector" type="checkbox"> sector Stage 2</label>
   <label class="tog"><input id="tDecline" type="checkbox"> after decline</label>
  </div>
  <button class="reset" id="reset">↺ validated values</button>
 </div>
 <div class="hint">Thresholds only: structural parameters (MA30, Mansfield lookback) can't be changed here.
  The "failed filters" slider opens the superset gradually: <b>0</b> = full only · <b>4</b> = everything.</div>
</div>

<div class="tabs" id="tabs"></div>
<div class="mktbar" id="mktbar"></div>
<div id="list"></div>
<div id="detail"></div>

<script>
const DATA = /*DATA*/;
const SC = DATA.stageColors, SN = DATA.stageNames, D = DATA.defaults;
let S = {...D}, maxFail = 1, curTab = null, sel = null, sortKey = "mansfield", sortDir = -1;

const $ = id => document.getElementById(id);
document.querySelector("#hdr").innerHTML =
  `Generated ${DATA.meta.generated} · data through <b>${DATA.meta.markets.map(m=>m.data_through).sort().slice(-1)[0]}</b>`
  + ` · ${DATA.meta.markets.reduce((a,m)=>a+m.universe,0).toLocaleString()} active stocks`;

// ---- valutazione client-side di un candidato contro le soglie correnti ----
function evalFails(c){
  const f = [];
  if (c.base_len < S.baseMin) f.push("base too short");
  if (c.base_depth_pct > S.depthMax) f.push("base too deep");
  if (c.vol_ratio < S.volMin) f.push("weak volume");
  if (c.mansfield < S.mansMin) f.push("Mansfield < min");
  if (c.risk_pct > S.riskMax) f.push("risk too high");
  if (S.reqMarket && c.f_market) f.push("market not in Stage 2");
  if (S.reqSector && c.f_sector) f.push("sector not in Stage 2");
  if (S.reqDecline && c.f_decline) f.push("base not after decline");
  for (const o of c.fixed_fails) f.push(o);
  return f;
}
const COLS = [
  ["kind","Type"],["fnow","Failed filter"],["ticker","Ticker"],["date","Signal"],
  ["entry","Entry"],["stop_bot","Stop"],["risk_pct","Risk"],["base_len","Base"],
  ["vol_ratio","Vol"],["mansfield","Mansfield"],["last_close","Last"]];

// filtro di vista: default = candidati "vicini" (falliscono ≤1 filtro alle soglie
// correnti) + pieni; "solo pieni" restringe; "mostra tutto" apre l'intero superset.
function keep(nfail){ return nfail <= maxFail; }   // 0 = solo pieni … superset_fails = tutto
function tabCount(m){ return m.candidates.filter(c=>keep(evalFails(c).length)).length; }
function marketRows(name){
  const m = DATA.markets.find(x=>x.name===name);
  let rows = m.candidates.map(c=>{const f=evalFails(c);return {...c, fnow:f, full:f.length===0};});
  rows = rows.filter(r=>keep(r.fnow.length));
  rows.sort((a,b)=>{
    if(a.full!==b.full) return a.full ? -1 : 1;   // i PIENI sempre in cima
    let va=a[sortKey], vb=b[sortKey];
    if(sortKey==="fnow"){va=a.fnow.length;vb=b.fnow.length;}
    if(va<vb)return -sortDir; if(va>vb)return sortDir; return 0;  // numerici: -1 = decrescente
  });
  return rows;
}
function renderTabs(){
  $("tabs").innerHTML = DATA.markets.map(m=>
    `<button class="tab ${m.name===curTab?'on':''}" data-m="${m.name}">${m.name}<span class="cnt">${tabCount(m)}</span></button>`
  ).join("");
  $("tabs").querySelectorAll(".tab").forEach(b=>b.onclick=()=>{curTab=b.dataset.m;sel=null;render();});
}
function renderList(){
  const m = DATA.markets.find(x=>x.name===curTab);
  const ph = m.stage===2 ? `<b style="color:var(--grn)">Stage 2 — buy</b>`
                         : `<b style="color:var(--amb)">Stage ${m.stage} — ${m.stage_name}</b>`;
  $("mktbar").innerHTML = `${m.name} · benchmark ${m.benchmark} · ${ph} · ${m.universe} active stocks · ${m.currency}`;
  const rows = marketRows(curTab);
  if(!rows.length){$("list").innerHTML=`<div class="none">No candidates with these thresholds.</div>`;return;}
  const th = COLS.map(([k,l])=>`<th data-k="${k}" class="${['entry','stop_bot','risk_pct','base_len','vol_ratio','mansfield','last_close'].includes(k)?'n':''}">${l}${sortKey===k?(sortDir<0?' ▼':' ▲'):''}</th>`).join("");
  const body = rows.map(r=>{
    const badge = r.full?'<span class="bp">FULL</span>':'<span class="bq">near</span>';
    return `<tr class="row ${sel===r.ticker?'sel':''}" data-tk="${r.ticker}">
      <td>${badge}</td><td class="fl">${r.fnow.join(', ')||'—'}</td>
      <td class="tk">${r.ticker}</td><td>${r.date}</td>
      <td class="n">${r.entry.toFixed(2)}</td><td class="n stop">${r.stop_bot.toFixed(2)}</td>
      <td class="n">${r.risk_pct.toFixed(1)}%</td><td class="n">${r.base_len}w</td>
      <td class="n">${r.vol_ratio.toFixed(2)}×</td><td class="n">${r.mansfield.toFixed(1)}</td>
      <td class="n">${r.last_close.toFixed(2)}</td></tr>`;
  }).join("");
  $("list").innerHTML = `<table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table>`;
  $("list").querySelectorAll("th").forEach(h=>h.onclick=()=>{
    const k=h.dataset.k; if(sortKey===k)sortDir=-sortDir; else{sortKey=k;sortDir=(k==='ticker'||k==='date')?1:-1;} render();});
  $("list").querySelectorAll("tr.row").forEach(tr=>tr.onclick=()=>{sel=tr.dataset.tk;render();});
}
// --- watchlist: seguire/smettere di seguire un candidato (POST al Worker) ---
function followKey(c){ return c.ticker+"|"+c.date; }
function followSet(){ try{return JSON.parse(localStorage.getItem("followed")||"{}");}catch(e){return {};} }
function isFollowed(c){ return !!followSet()[followKey(c)]; }
function updateFollowBtn(c,b){ const on=isFollowed(c); b.textContent=on?"✓ following":"★ follow"; b.className="follow"+(on?" on":""); }
async function toggleFollow(c,b){
  let secret=localStorage.getItem("watch_secret");
  if(!secret){ secret=prompt("Watchlist secret (once):"); if(!secret) return; localStorage.setItem("watch_secret",secret); }
  const action=isFollowed(c)?"unfollow":"follow"; b.disabled=true; b.textContent="…";
  try{
    const r=await fetch(DATA.watch_url.replace(/\/$/,"")+"/"+action,{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({secret,ticker:c.ticker,date:c.date,market:c.market,entry:c.entry,stop:c.stop_bot,base_len:c.base_len,mansfield:c.mansfield,vol_ratio:c.vol_ratio,currency:c.currency})});
    if(r.status===401){ alert("Wrong secret — reset it."); localStorage.removeItem("watch_secret"); b.disabled=false; updateFollowBtn(c,b); return; }
    const j=await r.json();
    if(j&&j.ok){ const st=followSet(),k=followKey(c); if(j.following)st[k]=true; else delete st[k]; localStorage.setItem("followed",JSON.stringify(st)); }
    else alert("Error: "+((j&&j.error)||"unknown"));
  }catch(e){ alert("Network error: "+e); }
  b.disabled=false; updateFollowBtn(c,b);
}
function renderDetail(){
  const d = $("detail");
  if(!sel){d.innerHTML="";return;}
  const m = DATA.markets.find(x=>x.name===curTab);
  const c = m.candidates.find(x=>x.ticker===sel);
  const f = evalFails(c), full = f.length===0;
  const badge = full?'<span class="bp">FULL</span>':`<span class="bq">NEAR · ${f.join(', ')}</span>`;
  // metriche con pass/fail rispetto alle soglie correnti
  const kv = [
    ["Entry",c.entry.toFixed(2),null],
    ["Stop",c.stop_bot.toFixed(2),null],
    ["Risk",c.risk_pct.toFixed(1)+'%', c.risk_pct<=S.riskMax],
    ["Base",c.base_len+'w', c.base_len>=S.baseMin],
    ["Depth",c.base_depth_pct.toFixed(0)+'%', c.base_depth_pct<=S.depthMax],
    ["Volume",c.vol_ratio.toFixed(2)+'×', c.vol_ratio>=S.volMin],
    ["Mansfield",c.mansfield.toFixed(1), c.mansfield>=S.mansMin],
    ["Shares",c.shares,null],["Position",c.position_eur.toLocaleString(),null],
  ].map(([k,v,ok])=>`<div class="${ok===true?'pass':ok===false?'fail':''}"><b>${k}</b>
      <span class="${ok===true?'ok':ok===false?'no':''}">${v}</span></div>`).join("");
  const fbtn = DATA.watch_url ? `<button class="follow" id="fbtn">★ follow</button>` : "";
  d.innerHTML = `<div class="card"><div class="chd">
     <span class="tk">${c.ticker}</span>${badge}
     <span class="meta">signal ${c.date} · ${c.weeks_ago}w ago · stage ${c.stage} · ${c.currency}</span>${fbtn}</div>
     <div id="chart"></div><div class="kv">${kv}</div></div>`;
  drawChart(c);
  if(DATA.watch_url && $("fbtn")){ updateFollowBtn(c, $("fbtn")); $("fbtn").onclick=()=>toggleFollow(c, $("fbtn")); }
}
function drawChart(c){
  const s = c.series;
  if(!s){$("chart").innerHTML='<div class="none">chart not available</div>';return;}
  const traces = [
    {type:"candlestick",x:s.t,open:s.o,high:s.h,low:s.l,close:s.c,name:"",
     increasing:{line:{color:"#2ca25f"}},decreasing:{line:{color:"#d6604d"}},xaxis:"x",yaxis:"y"},
    {type:"scatter",x:s.t,y:s.ma,mode:"lines",line:{color:"#2166ac",width:1.5},name:"MA30",xaxis:"x",yaxis:"y"},
    {type:"scatter",x:s.t,y:s.res,mode:"lines",line:{color:"#8c510a",width:1.2,dash:"dash"},name:"Resistance",xaxis:"x",yaxis:"y"},
    {type:"bar",x:s.t,y:s.vol,marker:{color:s.c.map((c2,i)=>c2>=s.o[i]?"#2ca25f":"#d6604d"),opacity:.6},name:"volume",xaxis:"x",yaxis:"y2"},
  ];
  // fasce di fase come rettangoli di sfondo
  const shapes=[]; let i0=0;
  for(let i=1;i<=s.stage.length;i++){
    if(i===s.stage.length||s.stage[i]!==s.stage[i0]){
      const col=SC[s.stage[i0]];
      if(col&&s.stage[i0]!==0) shapes.push({type:"rect",xref:"x",yref:"paper",
        x0:s.t[i0],x1:s.t[Math.min(i,s.t.length-1)],y0:0,y1:1,fillcolor:col,opacity:.28,line:{width:0},layer:"below"});
      i0=i;
    }
  }
  // entry + stop
  shapes.push({type:"line",xref:"x",yref:"y",x0:c.date,x1:s.t[s.t.length-1],y0:c.stop_bot,y1:c.stop_bot,
    line:{color:"#b2182b",width:1.4,dash:"dot"}});
  const ann=[
    {x:c.date,y:c.entry,xref:"x",yref:"y",text:"ENTRY "+c.entry.toFixed(2),showarrow:true,arrowcolor:"#00441b",
     arrowhead:6,ax:0,ay:-28,font:{color:"#0c8f4d",size:11}},
    {x:s.t[s.t.length-1],y:c.stop_bot,xref:"x",yref:"y",text:"STOP "+c.stop_bot.toFixed(2),showarrow:false,
     xanchor:"right",yanchor:"bottom",font:{color:"#d6604d",size:11}},
  ];
  const layout={paper_bgcolor:"#1a2128",plot_bgcolor:"#1a2128",font:{color:"#d7e0e6",size:11},
    margin:{l:48,r:16,t:10,b:28},showlegend:true,legend:{orientation:"h",y:1.02,x:0,font:{size:10}},
    xaxis:{domain:[0,1],rangeslider:{visible:false},gridcolor:"#222c35",anchor:"y2"},
    yaxis:{domain:[0.26,1],gridcolor:"#222c35",title:{text:"price"}},
    yaxis2:{domain:[0,0.2],gridcolor:"#222c35",title:{text:"vol"}},
    shapes,annotations:ann,dragmode:"zoom"};
  Plotly.newPlot("chart",traces,layout,{responsive:true,displayModeBar:true,scrollZoom:true,
    modeBarButtonsToRemove:["select2d","lasso2d"]});
}
function syncLabels(){
  $("lBase").textContent=S.baseMin;$("lDepth").textContent=S.depthMax;$("lVol").textContent=S.volMin.toFixed(1);
  $("lMans").textContent=S.mansMin;$("lRisk").textContent=S.riskMax;
  $("sBase").value=S.baseMin;$("sDepth").value=S.depthMax;$("sVol").value=S.volMin;
  $("sMans").value=S.mansMin;$("sRisk").value=S.riskMax;
  $("tMarket").checked=S.reqMarket;$("tSector").checked=S.reqSector;$("tDecline").checked=S.reqDecline;
  $("lMaxF").textContent=maxFail+(maxFail===0?" (full only)":maxFail>=4?" (all)":"");$("sMaxF").value=maxFail;
}
function render(){renderTabs();renderList();renderDetail();}
function bind(){
  const map={sBase:"baseMin",sDepth:"depthMax",sVol:"volMin",sMans:"mansMin",sRisk:"riskMax"};
  for(const [id,k] of Object.entries(map)) $(id).oninput=()=>{S[k]=parseFloat($(id).value);syncLabels();render();};
  $("tMarket").onchange=()=>{S.reqMarket=$("tMarket").checked;render();};
  $("tSector").onchange=()=>{S.reqSector=$("tSector").checked;render();};
  $("tDecline").onchange=()=>{S.reqDecline=$("tDecline").checked;render();};
  $("sMaxF").max = DATA.meta.superset_fails || 4;
  $("sMaxF").oninput=()=>{maxFail=parseInt($("sMaxF").value);syncLabels();render();};
  $("reset").onclick=()=>{S={...D};maxFail=1;syncLabels();render();};
}
curTab = (DATA.markets.find(m=>m.candidates.length)||DATA.markets[0]).name;
syncLabels();bind();render();
</script></body></html>
"""


if __name__ == "__main__":
    main()
    import os
    import sys
    sys.stdout.flush()          # pyarrow lascia thread → uscita forzata a lavoro finito
    sys.stderr.flush()
    os._exit(0)
