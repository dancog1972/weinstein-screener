#!/usr/bin/env python3
"""SCREENER settimanale: i candidati long di Weinstein di OGGI (e delle ultime
settimane), sui titoli ATTIVI di NASDAQ/NYSE.

Non e' un backtest: usa lo stesso motore validato (detect_signals + stop
strutturale su pivot) ma sui dati CORRENTI, e propone i candidati da valutare —
entry, stop suggerito, rischio per azione e size a rischio fisso. La decisione
finale e' umana: e' la direzione "scanner + giudizio" del progetto.

Uso:
    python screener.py                     # ultime 4 settimane, report HTML
    python screener.py --weeks 8 --top 30
    python screener.py --refresh           # riscarica il settimanale (post-chiusura)

Output: output/screener.html + output/screener.json
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from src.console import setup as _console_setup

_console_setup()

import numpy as np
import pandas as pd

from src.data_store import DataStore
from src.indicators import (confirmed_pivot_lows, enrich_weekly,
                            last_confirmed_pivot_before, to_weekly)
from src.config import load_config
from src.portfolio import Portfolio
from src.providers import make_provider
from src.signals import (Signal, detect_signals, initial_stop, prepare_ticker)
from src.stages import classify_stages
from src.universe import LiquidityUniverse

STAGE_NAME = {0: "—", 1: "Base", 2: "Avanzata", 3: "Top", 4: "Declino"}


def chart_b64(df: pd.DataFrame, sig, stop: float, weeks_back: int = 208) -> str:
    """Grafico del candidato: 4 anni di candele + MA30 + fasi + resistenza della
    base, con ENTRY e STOP segnati, e il pannello VOLUMI sotto (con la media a 4
    settimane: è il confronto che il metodo usa per validare il breakout).
    PNG base64 → il report HTML resta un file unico e autonomo (GitHub Pages)."""
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # riuso il disegno delle candele e delle fasi già scritto per make_charts
    from make_charts import STAGE_COLORS, contiguous_runs, draw_candles, fmt_price

    win = df.iloc[-weeks_back:] if len(df) > weeks_back else df   # ultimi 4 anni
    idx = win.index
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(11.5, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06})

    st = win["stage"].to_numpy()
    for code, (color, _lbl) in STAGE_COLORS.items():
        for a, b in contiguous_runs(st == code):
            left, right = idx[a], idx[min(b, len(idx) - 1)]
            ax.axvspan(left, right, color=color, alpha=0.5, linewidth=0, zorder=0)
            axv.axvspan(left, right, color=color, alpha=0.5, linewidth=0, zorder=0)
    draw_candles(ax, win)
    ax.plot(idx, win["ma"], color="#2166ac", lw=1.5, zorder=4, label="MA30")
    base_mask = pd.Series(win["base_len"].to_numpy() > 0, index=idx)
    ax.plot(idx, win["resistance"].where(base_mask), color="#8c510a", lw=1.2,
            ls="--", zorder=3, label="Resistenza base")

    # --- pannello VOLUMI: barre colorate come le candele + media 4 settimane ---
    vol = win["volume"].to_numpy(dtype=float)
    up = (win["adj_close"].to_numpy() >= win["adj_open"].to_numpy())
    axv.bar(idx, vol, width=4.2, color=np.where(up, "#2ca25f", "#d6604d"),
            alpha=0.75, zorder=2)
    vol_avg = win["volume"].shift(1).rolling(4, min_periods=4).mean()
    axv.plot(idx, vol_avg, color="#8c510a", lw=1.1, zorder=3, label="Media 4 sett.")
    # la settimana del segnale in risalto: è il volume che valida il breakout
    if sig.date in idx:
        axv.bar([sig.date], [float(win.loc[sig.date, "volume"])], width=4.2,
                color="#00441b", zorder=4)
    axv.set_ylabel("volume", fontsize=8)
    axv.legend(loc="upper left", fontsize=7, framealpha=0.9)
    axv.grid(True, axis="y", alpha=0.2)
    axv.tick_params(labelsize=8)
    axv.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _p: f"{v/1e6:.0f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))

    # ENTRY suggerito
    ax.scatter([sig.date], [sig.entry], marker="^", s=180, color="#00441b",
               edgecolors="white", linewidths=1.2, zorder=7)
    ax.annotate(f"ENTRY {fmt_price(sig.entry)}", (sig.date, sig.entry),
                textcoords="offset points", xytext=(7, 9), fontsize=9,
                fontweight="bold", color="#00441b")
    # STOP suggerito (strutturale)
    ax.hlines(stop, sig.date, idx[-1], color="#b2182b", lw=1.4, ls=":", zorder=5)
    ax.annotate(f"STOP {fmt_price(stop)}  (−{(sig.entry - stop) / sig.entry * 100:.1f}%)",
                (sig.date, stop), textcoords="offset points", xytext=(7, -13),
                fontsize=8.5, fontweight="bold", color="#b2182b")

    lo_y = min(float(win["adj_low"].min()), stop)
    ax.set_ylim(lo_y * 0.95, float(win["adj_high"].max()) * 1.05)
    # margine a destra: il segnale è recente (ultime settimane) e senza spazio
    # le etichette ENTRY/STOP finirebbero tagliate dal bordo
    ax.set_xlim(idx[0], idx[-1] + pd.Timedelta(weeks=16))
    ax.set_title(f"{sig.ticker} · segnale {sig.date.date()} · ultimi 4 anni",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.tick_params(labelbottom=False)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.08)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=92)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def active_universe(cfg: dict, universe_file: str, exchanges: list[str]) -> list[str]:
    """Solo i titoli ATTIVI oggi sulle borse indicate: un candidato delistato
    non si puo' comprare. Legge il parquet prodotto da `build_universe screen`."""
    up = Path(universe_file)
    if not up.exists():
        raise FileNotFoundError(f"universo non trovato: {up}\n"
                                f"  rigeneralo: python build_universe.py screen --exchange ...")
    u = pd.read_parquet(up)
    act = u[(~u["is_delisted"].astype(bool)) & (u["exchange_real"].isin(exchanges))]
    cache = {p.stem for p in (Path(cfg["data_dir"]) / cfg["provider"]).glob("*.parquet")}
    return sorted(t for t in act["ticker"] if t in cache)


def _build_quasi_signal(tk: str, df: pd.DataFrame, d, res_prev, ex_cfg: dict):
    """Costruisce il Signal di un QUASI-candidato (un breakout che fallisce
    qualche filtro). Applica le STESSE guardie anti-dati-corrotti del motore
    (detect_signals) e lo STESSO stop (initial_stop). Torna None se una guardia
    lo scarta — così un valore infinito/NaN non finisce mai nel report/JSON."""
    entry = float(df.loc[d, "adj_close"])
    rp = res_prev.loc[d]
    resistance = float(rp) if np.isfinite(rp) else None
    vr = float(df["vol_ratio"].loc[d])
    mans = float(df["mansfield"].loc[d])
    if not (np.isfinite(entry) and entry > 0):
        return None
    if not np.isfinite(vr) or vr > 100:
        return None
    if not np.isfinite(mans) or abs(mans) > 500:
        return None
    if resistance is None or resistance <= 0:
        return None
    stop = initial_stop(df, d, entry, resistance, ex_cfg)
    if not (stop < entry):
        return None
    return Signal(ticker=tk, date=d, entry=entry, stop=stop, resistance=resistance,
                  base_len=int(df["base_len"].shift(1).loc[d]),
                  base_depth=float(df["base_depth"].shift(1).loc[d]),
                  vol_ratio=vr, mansfield=mans)


def _row_from_signal(s, df, fails, pf, equity, name, curr, pivot_k) -> dict:
    """Riga del report a partire da un Signal (pieno o quasi). Il Signal arriva
    SEMPRE con valori finiti (detect_signals per i pieni, _build_quasi_signal per
    i quasi), quindi il JSON è sempre valido."""
    piv = last_confirmed_pivot_before(df, s.date, pivot_k)
    stop_pivot = round(float(piv), 2) if (piv is not None and piv < s.entry) else None
    shares = pf.size_fixed_risk(equity, s.entry, s.stop, side="long")
    row = df.loc[s.date]
    return {
        "kind": "pieno" if not fails else "quasi",
        "fails": fails,
        "ticker": s.ticker,
        "date": s.date.strftime("%Y-%m-%d"),
        "weeks_ago": int((df.index[-1] - s.date).days // 7),
        "entry": round(float(s.entry), 2),
        "stop_bot": round(float(s.stop), 2),
        "risk_pct": round((s.entry - s.stop) / s.entry * 100, 1),
        "stop_pivot": stop_pivot,
        "shares": int(shares),
        "position_eur": int(shares * s.entry),
        "base_len": int(s.base_len),
        "base_depth_pct": round(float(s.base_depth) * 100, 1),
        "vol_ratio": round(float(s.vol_ratio), 2),
        "mansfield": round(float(s.mansfield), 1),
        "stage": STAGE_NAME.get(int(row["stage"]), "—"),
        "last_close": round(float(df["adj_close"].iloc[-1]), 2),
        "market": name, "currency": curr,
        "_df": df, "_sig": s,     # per il grafico (rimossi prima del JSON)
    }


def scan_market(name: str, mspec: dict, cfg: dict, store, args,
                breakout_checks, bench_cache: dict, sector_map) -> tuple:
    """Scansiona UN mercato col suo benchmark e le sue soglie. Restituisce
    (candidati, info_mercato). I parametri del SEGNALE vengono da cfg (uguali per
    tutti i mercati); cambiano solo universo, benchmark e soglie di liquidità.

    PIENI = ciò che il motore validato (detect_signals) comprerebbe DAVVERO, poi
    filtrato per settore come nel backtest → parità totale, niente divergenze.
    QUASI = breakout che falliscono max_fails filtri, per il giudizio umano."""
    st_cfg, sig_cfg, ex_cfg = cfg["stages"], cfg["signal"], cfg["exits"]
    today = pd.Timestamp(args.asof) if getattr(args, "asof", None) else pd.Timestamp(date.today())
    start = (today - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    tickers = active_universe(cfg, mspec["universe_file"], mspec["exchanges"])
    # benchmark calcolato UNA volta per simbolo: DE/FR/AS/BR/LS condividono
    # EXSA.XETRA, non lo si ricalcola 5 volte.
    bsym = mspec["benchmark"]
    if bsym not in bench_cache:
        bd = store.get_with_warmup(bsym, start, end)
        bw = enrich_weekly(to_weekly(bd), None, st_cfg["ma_weeks"], st_cfg["slope_lookback"])
        bench_cache[bsym] = (bw, classify_stages(bw, st_cfg["flat_slope"]))
    bench, market_stage = bench_cache[bsym]
    mkt_now = int(market_stage.iloc[-1])
    print(f"\n[{name}] benchmark {bsym} · Fase {mkt_now} — {STAGE_NAME[mkt_now]}"
          f"{'  ✓ si compra' if mkt_now == 2 else '  ⚠ fuori Fase 2'}")
    print(f"[{name}] {len(tickers)} titoli attivi · {'/'.join(mspec['exchanges'])}")

    wkey = store.weekly_key(bsym, st_cfg["ma_weeks"],
                            st_cfg["slope_lookback"], st_cfg["flat_slope"], start, end)

    def _compute(daily):
        wk = enrich_weekly(to_weekly(daily), bench, st_cfg["ma_weeks"], st_cfg["slope_lookback"])
        return prepare_ticker(wk, st_cfg["flat_slope"])

    lq = mspec.get("liquidity", {})
    liq = LiquidityUniverse(
        min_price=lq.get("min_price", 5.0), min_volume=lq.get("min_volume", 200_000),
        min_dollar_volume=lq.get("min_dollar_volume", 5_000_000),
        lookback_weeks=lq.get("lookback_weeks", 13), max_volatility=lq.get("max_volatility"))

    pf_cfg = cfg["portfolio"]
    pf = Portfolio(pf_cfg["initial_capital"], pf_cfg["risk_per_trade"],
                   pf_cfg["max_position_pct"], pf_cfg["max_positions"],
                   pf_cfg["commission_bps"], pf_cfg["slippage_bps"])
    equity = float(pf_cfg["initial_capital"])
    curr = mspec.get("currency", "")
    pivot_k = ex_cfg.get("pivot_k", 2)

    def _sector_ok(tk, d):
        return sector_map is None or sector_map.sector_ok(tk, d)

    rows: list[dict] = []
    skipped = 0
    for i, tk in enumerate(tickers, 1):
        if i % 500 == 0:
            print(f"  [{name}] ...{i}/{len(tickers)} · {len(rows)} candidati finora", flush=True)
        try:
            df = store.get_weekly_enriched(tk, start, end, wkey, _compute).copy()
        except Exception:
            skipped += 1
            continue
        if df.empty or df["ma"].notna().sum() < st_cfg["ma_weeks"]:
            skipped += 1
            continue
        liq.compute(tk, df)
        cutoff = df.index[-min(args.weeks, len(df))]

        # --- PIENI: i segnali che il motore validato accetterebbe (stesse
        # guardie, stesso stop), poi il gate SETTORE come nel backtest. Un
        # ValueError qui è la rete di sicurezza anti-cache-stale: deve fallire
        # forte, non essere ingoiato (bug storico del progetto).
        sigs = detect_signals(tk, df, market_stage, sig_cfg, ex_cfg, st_cfg["flat_slope"])
        pieni = {s.date: s for s in sigs
                 if s.date >= cutoff and liq.is_eligible(tk, s.date) and _sector_ok(tk, s.date)}

        # --- QUASI: breakout recenti che falliscono 1..max_fails filtri. Gli
        # stessi filtri di detect_signals, valutati uno a uno (breakout_checks),
        # più il settore. Così un breakout bloccato da UN solo filtro non sparisce
        # in silenzio ma va al giudizio umano col motivo.
        try:
            core, checks = breakout_checks(df, market_stage, sig_cfg, st_cfg["flat_slope"])
        except Exception:
            core = checks = None
        res_prev = df["resistance"].shift(1)
        for d in df.index[df.index >= cutoff]:
            if d in pieni:
                s, fails = pieni[d], []
            else:
                if core is None or not bool(core.get(d, False)):
                    continue                   # nessuna rottura della resistenza
                if not liq.is_eligible(tk, d):
                    continue                   # non tradeabile per liquidità
                fails = [lbl for lbl, m in checks.items() if not bool(m.loc[d])]
                if not _sector_ok(tk, d):
                    fails.append("settore non in Fase 2")
                if not fails or len(fails) > args.superset_fails:
                    # 0 fail ma non è un pieno = scartato da una guardia dati o
                    # dallo stop (non un vero candidato). Oltre superset_fails =
                    # troppo lontano perfino per il superset esplorabile dagli slider.
                    continue
                s = _build_quasi_signal(tk, df, d, res_prev, ex_cfg)
                if s is None:
                    continue                   # guardia dati / stop non valido
            rows.append(_row_from_signal(s, df, fails, pf, equity, name, curr, pivot_k))

    info = {"name": name, "benchmark": bsym, "currency": curr,
            "stage": mkt_now, "stage_name": STAGE_NAME[mkt_now],
            "universe": len(tickers), "skipped": skipped,
            "data_through": bench.index[-1].strftime("%Y-%m-%d")}
    print(f"[{name}] {len(rows)} candidati grezzi")
    return rows, info


def load_sector_map(cfg: dict, store, args):
    """Carica la mappa settori (SPDR) e registra le serie di fase/RS degli ETF,
    come fa _prepare_backtest. Il filtro settore è il SECONDO schermo di Weinstein
    (mercato → settore → titolo) e il backtest lo applica: lo screener deve fare
    lo stesso, o i suoi 'pieni' non corrispondono a ciò che il metodo comprerebbe.

    La mappa copre i titoli USA (ETF SPDR in USD); per un ticker non mappato (es.
    europeo) sector_ok torna True → il filtro non si applica, esattamente come nel
    backtest. Torna None se sector_filter è off o manca la mappa (filtro inattivo,
    dichiarato)."""
    if not cfg["signal"].get("sector_filter", False):
        return None
    smap_path = next((spec.get("sector_map") for spec in cfg.get("universe", {}).values()
                      if spec.get("sector_map")), None)
    if not smap_path or not Path(smap_path).exists():
        print("  [settori] sector_filter=true ma manca sector_map: filtro INATTIVO")
        return None
    from src.sectors import SectorMap
    sm = SectorMap.from_csv(smap_path)
    today = pd.Timestamp(args.asof) if getattr(args, "asof", None) else pd.Timestamp(date.today())
    start = (today - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    st = cfg["stages"]
    us_bench = cfg.get("universe", {}).get("US", {}).get("benchmark", "SPY.US")
    b_wk = enrich_weekly(to_weekly(store.get_with_warmup(us_bench, start, end)),
                         None, st["ma_weeks"], st["slope_lookback"])
    for etf in sm.etfs_needed():
        try:
            w = enrich_weekly(to_weekly(store.get_with_warmup(etf, start, end)),
                              b_wk, st["ma_weeks"], st["slope_lookback"])
            sm.register_sector_series(etf, classify_stages(w, st["flat_slope"]), w["mansfield"])
        except Exception as e:  # noqa: BLE001
            print(f"  [settori] {etf} non disponibile: {e}")
    cov = sm.coverage()
    print(f"  [settori] {cov}")
    return sm if cov["etf_caricati"] else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Screener Weinstein — candidati long di oggi")
    ap.add_argument("--config", default="config_us_liquidity.yaml",
                    help="config coi parametri VALIDATI del segnale (stages/signal/exits/portfolio)")
    ap.add_argument("--markets", default="config_screener_markets.yaml",
                    help="config coi MERCATI (universo, benchmark, soglie per ciascuno)")
    ap.add_argument("--only", default=None, help="scansiona solo questo mercato (es. US, DE, FR)")
    ap.add_argument("--weeks", type=int, default=6, help="settimane indietro per i segnali")
    ap.add_argument("--years", type=int, default=6, help="storia caricata (per MA30/RS52)")
    ap.add_argument("--asof", default=None,
                    help="data di riferimento (YYYY-MM-DD); default oggi. Utile per girare "
                         "OFFLINE sull'ultima settimana in cache senza chiamate API.")
    ap.add_argument("--top", type=int, default=60, help="max candidati nel REPORT statico")
    ap.add_argument("--max-fails", type=int, default=1,
                    help="fail massimi di un QUASI nel REPORT statico (0 = solo pieni)")
    ap.add_argument("--superset-fails", type=int, default=4,
                    help="fail massimi nel SUPERSET dell'app interattiva (gli slider "
                         "esplorano qui: più alto = più margine per abbassare le soglie)")
    ap.add_argument("--superset-top", type=int, default=400,
                    help="tetto di righe nel superset dell'app (le migliori)")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    from make_charts import breakout_checks   # stessi filtri, valutati uno a uno

    import yaml
    cfg = load_config(args.config)
    # il file mercati ha solo `markets:`, non le sezioni obbligatorie di un config
    # completo → si legge con yaml semplice, non con load_config (che validerebbe).
    with open(args.markets, encoding="utf-8") as fh:
        markets = yaml.safe_load(fh)["markets"]   # dict: nome -> spec del mercato
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    print(f"Screener Weinstein · segnali delle ultime {args.weeks} settimane · "
          f"mercati: {', '.join(markets)}")

    sector_map = load_sector_map(cfg, store, args)   # secondo schermo (US · SPDR)
    bench_cache: dict = {}                            # benchmark calcolato 1 volta/simbolo
    all_rows: list[dict] = []
    infos: list[dict] = []
    for name, mspec in markets.items():
        if args.only and name != args.only:
            continue
        rows, info = scan_market(name, mspec, cfg, store, args, breakout_checks,
                                 bench_cache, sector_map)
        all_rows.extend(rows)
        infos.append(info)

    # un solo candidato per ticker: se un titolo rompe in più settimane recenti
    # (o è pieno una settimana e quasi in quella in corso, ancora parziale),
    # tengo il PIENO e, a parità, il più recente. Evita doppioni.
    best: dict[str, dict] = {}
    for r in all_rows:
        cur = best.get(r["ticker"])
        if cur is None or (r["kind"] == "pieno", r["date"]) > (cur["kind"] == "pieno", cur["date"]):
            best[r["ticker"]] = r
    # pieni prima dei quasi; dentro ciascun gruppo per forza relativa.
    superset = sorted(best.values(),
                      key=lambda r: (r["kind"] != "pieno", -r["mansfield"], r["risk_pct"]))
    superset = superset[: args.superset_top]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- REPORT statico: solo i candidati VALIDATI (fail ≤ max_fails), capped ---
    report_rows = [r for r in superset if len(r["fails"]) <= args.max_fails][: args.top]
    n_full = sum(1 for r in report_rows if r["kind"] == "pieno")
    if report_rows:
        print(f"\ndisegno {len(report_rows)} grafici...", flush=True)
    for r in report_rows:
        try:
            r["chart"] = chart_b64(r["_df"], r["_sig"], r["stop_bot"])
        except Exception as e:  # noqa: BLE001
            r["chart"] = None
            print(f"  [!] grafico {r['ticker']}: {e}")

    meta = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "markets": infos, "weeks_back": args.weeks,
            "candidates": len(report_rows), "full": n_full,
            "near": len(report_rows) - n_full, "max_fails": args.max_fails,
            "superset": len(superset), "superset_fails": args.superset_fails,
            "sector_active": sector_map is not None}

    # dati per il REPORT (validati) e per l'APP (superset): entrambi senza _df/_sig
    def _slim(rows, drop_chart):
        skip = {"_df", "_sig"} | ({"chart"} if drop_chart else set())
        return [{k: v for k, v in r.items() if k not in skip} for r in rows]

    (outdir / "screener.json").write_text(
        json.dumps({"meta": meta, "candidates": _slim(report_rows, True)}, indent=2),
        encoding="utf-8")
    # SUPERSET per l'app interattiva (le serie le aggiunge build_page): righe leggere
    (outdir / "screener_app_data.json").write_text(
        json.dumps({"meta": meta, "candidates": _slim(superset, True)}, ensure_ascii=False),
        encoding="utf-8")
    _write_html(outdir / "screener.html", meta, report_rows, cfg)

    print(f"\n{len(report_rows)} CANDIDATI VALIDATI ({n_full} pieni · "
          f"{len(report_rows)-n_full} quasi) · superset esplorabile: {len(superset)}")
    if report_rows:
        print(f"\n{'mkt':4s}{'ticker':12s}{'data':12s}{'entry':>8s}{'stop':>8s}"
              f"{'risk%':>7s}{'base':>6s}{'mans':>7s}  motivo")
        for r in report_rows[:25]:
            print(f"{r['market']:4s}{r['ticker']:12s}{r['date']:12s}{r['entry']:>8.2f}"
                  f"{r['stop_bot']:>8.2f}{r['risk_pct']:>7.1f}{r['base_len']:>6d}"
                  f"{r['mansfield']:>7.1f}  {', '.join(r['fails']) or '—'}")
    print(f"\n✓ Report: {outdir / 'screener.html'} · JSON: {outdir / 'screener.json'} · "
          f"App data: {outdir / 'screener_app_data.json'}")


def _write_html(path: Path, meta: dict, rows: list[dict], cfg: dict) -> None:
    """Report statico e autonomo (nessuna CDN): va bene anche su GitHub Pages."""
    # un banner per mercato: la Fase 2 abilita i long, altrimenti solo informativo
    banner = "".join(
        (f"<div class='ok'>{m['name']} ({m['benchmark']}): <b>Fase 2</b> — il filtro consente i long "
         f"· {m['universe']} titoli · dati al {m['data_through']}</div>"
         if m["stage"] == 2 else
         f"<div class='warn'>{m['name']} ({m['benchmark']}): <b>Fase {m['stage']} — {m['stage_name']}</b> "
         f"— fuori Fase 2, il metodo non compra · {m['universe']} titoli · dati al {m['data_through']}</div>")
        for m in meta["markets"])
    data_through = max((m["data_through"] for m in meta["markets"]), default="—")
    universe_tot = sum(m["universe"] for m in meta["markets"])
    trs = "".join(
        f"<tr><td>{'<span class=bp>PIENO</span>' if r['kind']=='pieno' else '<span class=bq>quasi</span>'}</td>"
        f"<td class='fl'>{', '.join(r['fails']) or '—'}</td>"
        f"<td><span class='mk'>{r['market']}</span></td>"
        f"<td class='tk'><a href='#c{i}'>{r['ticker']}</a></td><td>{r['date']}</td>"
        f"<td class='n'>{r['entry']:.2f}</td><td class='n stop'>{r['stop_bot']:.2f}</td>"
        f"<td class='n'>{r['risk_pct']:.1f}%</td>"
        f"<td class='n'>{r['stop_pivot'] if r['stop_pivot'] else '—'}</td>"
        f"<td class='n'>{r['shares']}</td><td class='n'>{r['position_eur']:,}</td>"
        f"<td class='n'>{r['base_len']}w</td><td class='n'>{r['base_depth_pct']:.0f}%</td>"
        f"<td class='n'>{r['vol_ratio']:.2f}×</td><td class='n'>{r['mansfield']:.1f}</td>"
        f"<td class='n'>{r['last_close']:.2f}</td></tr>"
        for i, r in enumerate(rows)) or "<tr><td colspan='16' class='none'>Nessun candidato in questo periodo.</td></tr>"

    # una SCHEDA per candidato: grafico con entry/stop segnati + i numeri
    cards = "".join(
        f"""<div class="card" id="c{i}">
 <div class="chd"><span class="mk">{r['market']}</span><span class="tk">{r['ticker']}</span>
  {'<span class="bp">PIENO</span>' if r['kind'] == 'pieno' else '<span class="bq">QUASI · ' + ', '.join(r['fails']) + '</span>'}
  <span class="meta">segnale {r['date']} · {r['weeks_ago']} sett. fa · fase {r['stage']} · {r['currency']}</span></div>
 {'<img src="data:image/png;base64,' + r['chart'] + '">' if r.get('chart') else '<div class="none">grafico non disponibile</div>'}
 <div class="kv">
  <div><b>Entry</b><span>{r['entry']:.2f}</span></div>
  <div><b>Stop</b><span class="stop">{r['stop_bot']:.2f}</span></div>
  <div><b>Rischio</b><span>{r['risk_pct']:.1f}%</span></div>
  <div><b>Azioni</b><span>{r['shares']}</span></div>
  <div><b>Posizione</b><span>{r['position_eur']:,}</span></div>
  <div><b>Base</b><span>{r['base_len']}w · {r['base_depth_pct']:.0f}%</span></div>
  <div><b>Volume</b><span>{r['vol_ratio']:.2f}×</span></div>
  <div><b>Mansfield</b><span>{r['mansfield']:.1f}</span></div>
  <div><b>Ultimo</b><span>{r['last_close']:.2f}</span></div>
 </div></div>"""
        for i, r in enumerate(rows))
    s = cfg["signal"]
    html = f"""<!doctype html><meta charset="utf-8"><title>Screener Weinstein · {meta['generated'][:10]}</title>
<style>
 body{{background:#151b21;color:#d7e0e6;font:13px/1.5 -apple-system,Segoe UI,sans-serif;margin:0;padding:24px}}
 h1{{font-size:19px;margin:0 0 4px}} .sub{{color:#7f8c98;font-size:12px;margin-bottom:14px}}
 .ok{{background:#16341f;border-left:3px solid #6fe3a1;padding:8px 12px;border-radius:4px;margin:12px 0}}
 .warn{{background:#3a2a16;border-left:3px solid #e0a458;padding:8px 12px;border-radius:4px;margin:12px 0}}
 table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:12px}}
 th{{text-align:left;color:#7f8c98;font-weight:600;border-bottom:1px solid #2b353f;padding:7px 8px;white-space:nowrap}}
 td{{border-bottom:1px solid #222c35;padding:7px 8px}} tr:hover td{{background:#1a2128}}
 .n{{text-align:right;font-family:SF Mono,Consolas,monospace}} .tk{{font-weight:600;color:#6fe3a1}}
 .tk a{{color:#6fe3a1;text-decoration:none}} .tk a:hover{{text-decoration:underline}}
 .stop{{color:#d6604d}} .none{{text-align:center;color:#7f8c98;padding:24px}}
 .foot{{color:#7f8c98;font-size:11px;margin-top:16px;line-height:1.7}}
 h2{{font-size:15px;margin:26px 0 10px;color:#d7e0e6}}
 .card{{background:#1a2128;border:1px solid #2b353f;border-radius:8px;padding:12px;margin-bottom:18px}}
 .card img{{width:100%;height:auto;border-radius:4px;display:block}}
 .chd{{display:flex;align-items:baseline;gap:10px;margin-bottom:8px}}
 .chd .tk{{font-size:16px}} .chd .meta{{color:#7f8c98;font-size:11px}}
 .kv{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}}
 .kv div{{background:#151b21;border-radius:4px;padding:5px 10px;min-width:78px}}
 .kv b{{display:block;color:#7f8c98;font-size:10px;font-weight:600;text-transform:uppercase}}
 .kv span{{font-family:SF Mono,Consolas,monospace;font-size:13px}}
 .bp{{background:#16341f;color:#6fe3a1;border:1px solid #2f6b45;border-radius:3px;
      padding:1px 6px;font-size:10px;font-weight:700}}
 .bq{{background:#3a2a16;color:#e0a458;border:1px solid #6b5330;border-radius:3px;
      padding:1px 6px;font-size:10px;font-weight:600}}
 .fl{{color:#e0a458;font-size:11px}}
 .mk{{background:#233240;color:#8fb8d8;border-radius:3px;padding:1px 6px;font-size:10px;font-weight:700}}
</style>
<h1>Screener Weinstein — candidati long</h1>
<div class="sub">Generato {meta['generated']} · dati al <b>{data_through}</b> ·
 universo {universe_tot} titoli attivi · segnali delle ultime {meta['weeks_back']} settimane</div>
{banner}
<div class="sub"><b>{meta['full']}</b> candidati pieni (passano tutti i filtri) ·
 <b>{meta['near']}</b> quasi-candidati (falliscono max {meta['max_fails']} filtro — <i>giudichi tu</i>)</div>
<table>
<tr><th>Tipo</th><th>Filtro fallito</th><th>Mkt</th><th>Ticker</th><th>Segnale</th><th>Entry</th><th>Stop</th><th>Rischio</th><th>Stop pivot</th>
<th>Azioni</th><th>Posizione</th><th>Base</th><th>Prof.</th><th>Vol</th><th>Mansfield</th><th>Ultimo</th></tr>
{trs}
</table>
<h2>Candidati — grafico con ingresso e stop</h2>
{cards}
<div class="foot">
<b>Come leggerlo.</b> <b>Entry</b> = chiusura del breakout; l'esecuzione reale va all'apertura successiva.
<b>Stop</b> = strutturale, sotto l'ultimo minimo confermato (−{cfg['exits'].get('pivot_buffer',0.02)*100:.0f}% di buffer).
<b>Azioni/Posizione</b> = size a rischio fisso ({cfg['portfolio']['risk_per_trade']*100:.0f}% del capitale
di {cfg['portfolio']['initial_capital']:,}) — ricalcola sul TUO capitale.
<b>Mansfield</b> = forza relativa vs mercato (richiesto ≥ {s['mansfield_min']:.0f} e in salita).<br>
<b>Filtri attivi</b>: base ≥ {s['min_base_weeks']}w, profondità ≤ {s['max_base_depth']*100:.0f}%,
volume ≥ {s['volume_ratio_min']}×, dopo un declino, contrazione volume o accumulo (OBV),
niente breakout da notizia (&gt;{s.get('max_breakout_stretch',0)*100:.0f}% sopra la resistenza), mercato in Fase 2{
' e settore in Fase 2 (US, ETF SPDR; i titoli EU non hanno mappa settoriale → filtro non applicato)' if meta.get('sector_active') else ' (filtro settore non attivo)'}.<br>
<b>Ordinati per Mansfield</b> (forza relativa) decrescente. Lo screener <i>propone</i>: la selezione finale è tua.
</div>"""
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
    import os
    import sys
    sys.stdout.flush()          # pyarrow lascia thread → uscita forzata a lavoro finito
    sys.stderr.flush()
    os._exit(0)
