"""Backtest automatico: il bot applica le regole Weinstein senza discrezione.

Struttura pensata per il deliverable 2 (simulatore human-vs-bot): il loop
settimanale è identico, cambia solo CHI decide. Qui decide `bot_decide`.

Anti look-ahead: il segnale nasce alla chiusura del venerdì t; l'esecuzione
avviene sulla stessa chiusura con slippage (approssimazione prudente della
apertura del lunedì). Gli stop sono verificati sulla chiusura settimanale,
in coerenza con la natura settimanale del metodo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import time as _time

import numpy as np
import pandas as pd

from .data_store import DataStore
from .indicators import confirmed_pivot_lows, enrich_weekly, to_weekly
from .portfolio import Portfolio
from .signals import Signal, detect_signals, forward_returns, prepare_ticker
from .stages import classify_stages


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    bench_curve: pd.Series
    trades: list
    signals: list[Signal]
    signal_stats: pd.DataFrame          # una riga per segnale, con fwd/exc returns
    metrics: dict[str, Any]
    per_ticker: dict[str, pd.DataFrame] = field(default_factory=dict)
    market_stage: pd.Series | None = None


# ----------------------------------------------------------------------
def load_universe(cfg: dict) -> dict[str, dict]:
    """Risolve l'universo effettivo per ogni mercato.

    Tre modalità, dichiarate in config con `universe_mode`:
      - `list`       : lista esplicita di ticker (o synthetic)
      - `index_pit`  : composizione STORICA dell'indice (point-in-time).
                       Elimina il survivorship bias: un titolo è comprabile
                       solo nelle settimane in cui era davvero nell'indice.
      - `index_static`: composizione ATTUALE dell'indice applicata a tutto il
                       periodo. Comodo ma con survivorship bias: da usare solo
                       dove i dati storici non esistono (es. indici europei),
                       e da DICHIARARE nel report.
    """
    from .providers.synthetic import SyntheticProvider
    from .universe import LiquidityUniverse, PointInTimeUniverse

    uni = {}
    for market, spec in cfg["universe"].items():
        mode = spec.get("mode", "list")
        pit = liq = None
        if mode == "liquidity":
            # universo definito da una REGOLA sui dati: point-in-time gratis,
            # nessun bisogno di dati fondamentali a pagamento
            lq = spec.get("liquidity", {})
            liq = LiquidityUniverse(
                min_price=lq.get("min_price", 5.0),
                min_volume=lq.get("min_volume", 200_000),
                min_dollar_volume=lq.get("min_dollar_volume", 5_000_000),
                lookback_weeks=lq.get("lookback_weeks", 13),
                max_volatility=lq.get("max_volatility"),   # None = disattivato
                atr_weeks=lq.get("atr_weeks", 14))
            tickers = list(spec.get("tickers") or [])
            if not tickers:
                # nessuna lista esplicita: la si prende dal parquet costruito
                # da `build_universe.py screen`. Scaricare gli storici e poi
                # dover incollare 800 ticker a mano nel config sarebbe assurdo.
                up = spec.get("universe_file")
                if not up:
                    up = Path(cfg["data_dir"]) / f"universe_{market}.parquet"
                up = Path(up)
                if not up.exists():
                    raise FileNotFoundError(
                        f"Universo per liquidità vuoto e file non trovato: {up}\n"
                        f"  Costruiscilo prima:\n"
                        f"    python build_universe.py screen --exchange {market}\n"
                        f"  oppure indica `universe_file:` nel config."
                    )
                tickers = sorted(pd.read_parquet(up)["ticker"].tolist())
                # si tengono solo quelli davvero scaricati: uno screen più ampio
                # del download non deve far fallire il backtest
                cache = Path(cfg["data_dir"]) / cfg["provider"]
                if cache.exists():
                    have = {p.stem for p in cache.glob("*.parquet")}
                    missing = [t for t in tickers if t not in have]
                    tickers = [t for t in tickers if t in have]
                    if missing:
                        print(f"  [universo] {len(missing)} ticker senza prezzi in cache, "
                              f"esclusi (scaricali con `prices --from-universe`)")
        elif mode in ("index_pit", "index_static"):
            index = spec.get("index", spec["benchmark"])
            if mode == "index_pit":
                pit = PointInTimeUniverse.load(cfg["data_dir"], index)
                tickers = pit.all_tickers()
            else:
                tickers = list(spec.get("tickers") or [])
                pit = PointInTimeUniverse.static(tickers, index)
        else:
            tickers = list(spec.get("tickers") or [])
            if not tickers and cfg["provider"] == "synthetic":
                tickers = SyntheticProvider().universe()
        uni[market] = {"benchmark": spec["benchmark"], "tickers": tickers,
                       "mode": mode, "pit": pit, "liq": liq,
                       "sector_map": spec.get("sector_map")}
    return uni


def _prepare_backtest(cfg, store, universe, start, end, horizons, st_cfg, progress):
    """Fase PESANTE del backtest: carica benchmark, settori e titoli, calcola
    i segnali e le statistiche. Non dipende dal rischio di portafoglio, quindi
    può essere calcolata una volta e riusata per più livelli di rischio
    (comando `experiment`). Restituisce tutto ciò che serve alla simulazione."""
    bench_data: dict = {}
    market_stage: dict = {}
    for market, spec in universe.items():
        b_daily = store.get_with_warmup(spec["benchmark"], start, end)
        b_wk = enrich_weekly(to_weekly(b_daily), None, st_cfg["ma_weeks"], st_cfg["slope_lookback"])
        bench_data[market] = b_wk
        market_stage[market] = classify_stages(b_wk, st_cfg["flat_slope"])

    # --- 1b. Settori: il secondo schermo del triple screen di Weinstein ---
    # La forza del settore è misurata sugli ETF SPDR con lo stesso motore di
    # fasi usato per i titoli. Se manca la mappa, il filtro è inattivo.
    sector_map = None
    if cfg["signal"].get("sector_filter", False):
        from .sectors import SectorMap
        first = next(iter(universe.values()))
        smap_path = first.get("sector_map")
        if smap_path:
            sector_map = SectorMap.from_csv(smap_path)
            bench0 = bench_data[next(iter(universe))]
            for etf in sector_map.etfs_needed():
                try:
                    d = store.get_with_warmup(etf, start, end)
                    w = enrich_weekly(to_weekly(d), bench0,
                                      st_cfg["ma_weeks"], st_cfg["slope_lookback"])
                    sector_map.register_sector_series(
                        etf, classify_stages(w, st_cfg["flat_slope"]), w["mansfield"])
                except Exception as e:  # noqa: BLE001
                    if progress:
                        print(f"  [settori] {etf} non disponibile: {e}")
            if progress:
                print(f"  [settori] {sector_map.coverage()}")
        elif progress:
            print("  [settori] sector_filter=true ma manca sector_map: filtro INATTIVO")

    per_ticker: dict[str, pd.DataFrame] = {}
    ticker_market: dict[str, str] = {}
    all_signals: list[Signal] = []
    skipped_nodata = 0            # ticker senza dati validi: saltati, non fatali
    for market, spec in universe.items():
        # chiave della cache settimanale: dipende dai parametri di STAGE e dal
        # benchmark, non dalle soglie di segnale. Cambiare volume_ratio_min o
        # gli stop riusa la cache; cambiare la MA30 o il benchmark la rigenera.
        wkey = store.weekly_key(spec["benchmark"], st_cfg["ma_weeks"],
                                st_cfg["slope_lookback"], st_cfg["flat_slope"],
                                start, end)

        def _compute_weekly(daily, _mkt=market):
            wk = enrich_weekly(to_weekly(daily), bench_data[_mkt],
                               st_cfg["ma_weeks"], st_cfg["slope_lookback"])
            return prepare_ticker(wk, st_cfg["flat_slope"])

        for tk in spec["tickers"]:
            # Su 20.000 titoli, alcuni hanno cache vuota o storico troppo corto
            # per far maturare la MA30. Uno di questi non deve far crollare
            # l'intero backtest: lo si salta e si tiene il conto.
            try:
                df = store.get_weekly_enriched(tk, start, end, wkey, _compute_weekly)
            except Exception:  # noqa: BLE001
                skipped_nodata += 1
                continue
            if df.empty or df["ma"].notna().sum() < st_cfg["ma_weeks"]:
                skipped_nodata += 1
                continue
            df["swing_low"] = confirmed_pivot_lows(df, cfg["exits"].get("pivot_k", 2))
            if spec.get("liq") is not None:
                spec["liq"].compute(tk, df)      # idoneità settimana per settimana
            per_ticker[tk] = df
            ticker_market[tk] = market
            sigs = detect_signals(tk, df, market_stage[market], cfg["signal"],
                                  cfg["exits"], st_cfg["flat_slope"])
            sigs = [s for s in sigs if pd.Timestamp(start) <= s.date <= pd.Timestamp(end)]
            all_signals.extend(sigs)
            # su 20.000 titoli una riga per ognuno intasa il terminale:
            # progresso ogni 1000, e solo se ci sono molti ticker
            if progress and len(spec["tickers"]) > 200:
                done = len(per_ticker) + skipped_nodata
                if done % 1000 == 0:
                    print(f"  ...{done}/{len(spec['tickers'])} titoli · "
                          f"{len(all_signals)} segnali finora", flush=True)
            elif progress:
                print(f"  {tk}: {len(sigs)} segnali")
    all_signals.sort(key=lambda s: s.date)

    # --- 2. Statistiche per segnale (solo segnali TRADEABILI) ------------
    # signal_stats alimenta l'analisi di qualità e la scelta dei casi da
    # graficare. Deve riflettere ciò che la strategia comprerebbe DAVVERO: un
    # breakout su un titolo sotto la soglia di liquidità a quella data (sub-penny,
    # illiquido) non è un segnale operativo, è rumore. Lo escludiamo con lo STESSO
    # criterio del portafoglio (liq.is_eligible), altrimenti i "migliori/peggiori"
    # finiscono dominati da penny-stock mai negoziabili. all_signals resta intero
    # per la simulazione (che applica comunque il proprio filtro di liquidità).
    def _tradeable(s: Signal) -> bool:
        liq = universe[ticker_market[s.ticker]].get("liq")
        return liq is None or liq.is_eligible(s.ticker, s.date)

    rows = []
    for s in all_signals:
        if not _tradeable(s):
            continue
        bench = bench_data[ticker_market[s.ticker]]
        fr = forward_returns(s, per_ticker[s.ticker], bench, horizons)
        rows.append({"ticker": s.ticker, "date": s.date, "entry": s.entry,
                     "stop": s.stop, "base_len": s.base_len,
                     "base_depth": s.base_depth, "vol_ratio": s.vol_ratio,
                     "mansfield": s.mansfield, **fr})
    signal_stats = pd.DataFrame(rows)
    return (bench_data, market_stage, sector_map, per_ticker, ticker_market,
            all_signals, signal_stats, skipped_nodata)


def run_backtest(cfg: dict, store: DataStore, progress: bool = False) -> BacktestResult:
    """Backtest completo su un universo. La fase pesante (scansione titoli e
    segnali) è isolata in _prepare_backtest, la simulazione di portafoglio
    segue. Un solo livello di rischio per chiamata."""
    bt, st_cfg = cfg["backtest"], cfg["stages"]
    start, end = bt["start"], bt["end"]
    horizons = bt["horizons_weeks"]
    universe = load_universe(cfg)

    # --- 1. Fase pesante: benchmark, settori, titoli, segnali ------------
    bench_data, market_stage, sector_map, per_ticker, ticker_market, \
        all_signals, signal_stats, skipped_nodata = _prepare_backtest(
            cfg, store, universe, start, end, horizons, st_cfg, progress)


    # --- 3. Simulazione di portafoglio (loop settimanale) ----------------
    pf_cfg = cfg["portfolio"]
    pf = Portfolio(pf_cfg["initial_capital"], pf_cfg["risk_per_trade"],
                   pf_cfg["max_position_pct"], pf_cfg["max_positions"],
                   pf_cfg["commission_bps"], pf_cfg["slippage_bps"])

    # calendario = unione delle settimane di tutti i titoli nel periodo
    weeks = sorted(set().union(*[set(df.loc[start:end].index) for df in per_ticker.values()]))
    sig_by_week: dict[pd.Timestamp, list[Signal]] = {}
    for s in all_signals:
        sig_by_week.setdefault(s.date, []).append(s)

    # --- SHORT: segnali di breakdown (Fase 4), speculari al long. per_ticker ha
    # già le feature del top. Idoneità di liquidità come per il long. ----------
    enable_short = cfg.get("short", {}).get("enabled", False)
    # ribilanciamento: se pienamente investito e arriva un nuovo segnale long,
    # si fa spazio vendendo una fetta proporzionale delle long esistenti.
    rebalance_on_signal = pf_cfg.get("rebalance_on_signal", False)
    sig_by_week_short: dict[pd.Timestamp, list[Signal]] = {}
    n_short_sig = 0
    if enable_short:
        from .signals import detect_signals_short
        # inietto la conferma dell'orso (parametro dello short) nel sig_cfg
        short_sig_cfg = {**cfg["signal"],
                         "bear_confirm_weeks": cfg.get("short", {}).get("bear_confirm_weeks", 0)}
        for tk, df in per_ticker.items():
            liq = universe[ticker_market[tk]].get("liq")
            for s in detect_signals_short(tk, df, market_stage[ticker_market[tk]],
                                          short_sig_cfg, cfg["exits"], st_cfg["flat_slope"]):
                if not (pd.Timestamp(start) <= s.date <= pd.Timestamp(end)):
                    continue
                if liq is not None and not liq.is_eligible(tk, s.date):
                    continue
                sig_by_week_short.setdefault(s.date, []).append(s)
                n_short_sig += 1
        if progress:
            print(f"  [short] {n_short_sig} segnali di breakdown (Fase 4)")

    # appartenenza point-in-time: ticker -> {settimana: era nell'indice?}
    # Solo per i mercati in modalità index_pit; altrove la mappa resta vuota
    # e il filtro non scatta (comportamento invariato).
    ticker_pit: dict[str, dict] = {}
    weeks_idx = pd.DatetimeIndex(weeks)
    for market, spec in universe.items():
        pit = spec.get("pit")
        if pit is None or spec.get("mode") != "index_pit":
            continue
        mat = pit.membership_matrix(weeks_idx)
        for tk in mat.columns:
            if tk in per_ticker:
                ticker_pit[tk] = mat[tk].to_dict()
    skipped_pit = skipped_liq = skipped_sector = 0
    forced_delist = 0
    # valutazione del delisting: "zero" (fallimento) o "last_price" (default)
    delist_zero = cfg["exits"].get("delisting_value", "last_price") == "zero"

    # ultima settimana con dati per ogni titolo: oltre quella data il titolo
    # NON esiste più (delistato, fuso, fallito) e non può essere valutato
    last_bar = {tk: df.index[-1] for tk, df in per_ticker.items()}

    equity_hist = []
    exposure_hist: list[float] = []
    npos_hist: list[int] = []
    _t_sim = _time.time()
    _n_weeks = len(weeks)
    for _wi, wk_date in enumerate(weeks):
        if progress and _n_weeks > 200 and _wi and _wi % 100 == 0:
            _el = _time.time() - _t_sim
            _eta = _el / _wi * (_n_weeks - _wi)
            print(f"  [portafoglio] {_wi}/{_n_weeks} settimane "
                  f"({_wi/_n_weeks*100:.0f}%) · {len(pf.positions)} posizioni "
                  f"aperte · ~{_eta/60:.0f} min rimanenti", flush=True)
        # Prezzi correnti. `asof()` propagherebbe l'ultimo prezzo noto ALL'INFINITO:
        # un titolo delistato nel 2008 varrebbe ancora il suo ultimo prezzo nel 2024.
        # Con l'universo per liquidità (2/3 delistati) questo gonfierebbe l'equity
        # con posizioni fantasma. Valutiamo solo i titoli ancora vivi.
        prices = {}
        for tk, df in per_ticker.items():
            if wk_date > last_bar[tk]:
                continue                       # titolo non più quotato
            p = df["adj_close"].asof(wk_date)
            if p == p:                         # non NaN
                prices[tk] = float(p)

        # 3a-bis. DELISTING: se un titolo in portafoglio ha esaurito i dati,
        # la posizione va chiusa all'ultimo prezzo disponibile. Non farlo
        # significa tenere aperta una posizione su un'azienda che non esiste,
        # occupando uno slot di `max_positions` per il resto del backtest.
        for tk in list(pf.positions):
            if wk_date > last_bar[tk]:
                last_px = (0.0 if delist_zero
                           else float(per_ticker[tk]["adj_close"].iloc[-1]))
                pf.close(tk, last_bar[tk], last_px, "delisting")
                forced_delist += 1

        pf.mark_prices(prices)      # aggiorna last_price: equity() resta onesta

        # 3a. trailing dello stop. Due modi (config `trailing_mode`):
        #   "swing" = sotto ogni minimo di pullback CONFERMATO (stretto, appeso al
        #             singolo pivot: esce anche sugli storni di Fase 2).
        #   "ma30"  = segue la MA30 (stop = MA30*(1-buffer), solo in salita): si
        #             tiene per tutta la Fase 2 finché il prezzo non chiude sotto la
        #             media — fedele al "tenere fino alla Fase 4" di Weinstein, e non
        #             fragile rispetto a QUALE pivot (il problema visto su INFN/MDSO).
        # Retrocompatibilità: senza `trailing_mode`, si usa il vecchio `trailing_swing`.
        mode = cfg["exits"].get("trailing_mode")
        if mode is None:
            mode = "swing" if cfg["exits"].get("trailing_swing", False) else "none"
        if mode == "swing":
            k = cfg["exits"].get("pivot_k", 2)
            buf = cfg["exits"].get("swing_buffer", 0.03)
            for tk, pos in pf.positions.items():
                if pos.side != "long":            # swing è solo long
                    continue
                df = per_ticker[tk]
                if wk_date not in df.index:
                    continue
                sw = df.loc[wk_date, "swing_low"]
                if np.isnan(sw):
                    continue
                pivot_week = df.index[df.index.get_loc(wk_date) - k]
                if pivot_week <= pos.entry_date:
                    continue                      # pivot della base, non del trend
                new_stop = float(sw) * (1.0 - buf)
                if new_stop > pos.stop:           # lo stop può solo salire
                    pos.stop = new_stop
                    pos.raised = True
        elif mode == "ma30":
            # buffer dalla MA30 legato all'ATR (k×ATR, adattivo alla volatilità).
            # LONG: stop = MA30 − dist, sale. SHORT (specchio): stop = MA30 + dist,
            # scende. In entrambi lo stop si muove solo VERSO il prezzo.
            k_atr = cfg["exits"].get("ma_trail_atr_mult")
            buf = cfg["exits"].get("ma_trail_buffer", 0.05)
            for tk, pos in pf.positions.items():
                df = per_ticker[tk]
                if wk_date not in df.index:
                    continue
                ma = df.loc[wk_date, "ma"]
                if np.isnan(ma):
                    continue
                atr = df.loc[wk_date, "atr"] if "atr" in df.columns else np.nan
                dist = (k_atr * float(atr)) if (k_atr is not None and np.isfinite(atr)) \
                    else float(ma) * buf
                if pos.side == "long":
                    new_stop = float(ma) - dist
                    if new_stop > pos.stop:       # long: lo stop può solo salire
                        pos.stop = new_stop
                        pos.raised = True
                else:                             # short: lo stop può solo scendere
                    new_stop = float(ma) + dist
                    if new_stop < pos.stop:
                        pos.stop = new_stop
                        pos.raised = True

        # 3b. uscite. STOP: valutato sul MINIMO della settimana, non sulla
        # chiusura. Un ordine di stop reale sul mercato scatta quando il prezzo
        # LO TOCCA, non aspetta il venerdì: valutarlo sulla chiusura sottostima
        # le perdite (il prezzo può crollare mercoledì e recuperare venerdì, ma
        # il tuo stop sarebbe già scattato). Con stop_intraweek=True (default,
        # realistico) usciamo ALLO STOP se adj_low lo ha toccato.
        # MA30 BREAKDOWN: resta sulla chiusura. Non è un ordine sul mercato ma
        # un segnale di tendenza, che Weinstein valuta a fine settimana.
        # costo di prestito settimanale sui corti (carry dello short)
        borrow_weekly = cfg.get("short", {}).get("borrow_annual", 0.0) / 52.0
        pf.charge_borrow(prices, borrow_weekly)

        stop_intraweek = cfg["exits"].get("stop_intraweek", True)
        for tk in list(pf.positions):
            df = per_ticker[tk]
            if wk_date not in df.index:
                continue
            row = df.loc[wk_date]
            pos = pf.positions[tk]
            close = float(row["adj_close"])
            ma = row["ma"]
            slope = row["ma_slope"]
            if pos.side == "long":
                low = float(row.get("adj_low", close))
                touched = (low <= pos.stop) if stop_intraweek else (close <= pos.stop)
                if touched:
                    exit_px = pos.stop
                    open_px = float(row.get("adj_open", close))
                    if open_px < pos.stop:        # gap sotto: esce all'apertura
                        exit_px = open_px
                    pf.close(tk, wk_date, exit_px,
                             "trailing_stop" if pos.raised else "stop")
                elif cfg["exits"]["ma_breakdown"] and not np.isnan(ma) \
                        and close < ma and slope < 0:
                    pf.close(tk, wk_date, close, "ma_breakdown")
            else:                                 # SHORT: tutto specchiato
                high = float(row.get("adj_high", close))
                touched = (high >= pos.stop) if stop_intraweek else (close >= pos.stop)
                if touched:
                    exit_px = pos.stop
                    open_px = float(row.get("adj_open", close))
                    if open_px > pos.stop:        # gap sopra: ricopre all'apertura
                        exit_px = open_px
                    pf.close(tk, wk_date, exit_px,
                             "trailing_stop" if pos.raised else "stop")
                elif cfg["exits"]["ma_breakdown"] and not np.isnan(ma) \
                        and close > ma and slope > 0:   # rientro sopra MA30 in salita
                    pf.close(tk, wk_date, close, "ma_breakup")

        # 3c. ingressi. TIMING DI ESECUZIONE (anti look-ahead di esecuzione):
        # il segnale si constata alla CHIUSURA della settimana t (il breakout è
        # adj_close[t] > resistenza). Comprare a adj_close[t] significherebbe
        # eseguire al prezzo stesso che ha rivelato il segnale — impossibile
        # nella realtà. Con execution="next_open" (default) l'ordine si esegue
        # all'APERTURA della settimana t+1: è ciò che potresti fare davvero
        # vedendo il breakout a mercato chiuso. "same_close" mantiene il vecchio
        # comportamento (ottimistico), utile solo per confronto.
        execution = cfg["exits"].get("execution", "next_open")
        for s in sig_by_week.get(wk_date, []):
            pit = ticker_pit.get(s.ticker)
            if pit is not None and not pit.get(wk_date, False):
                skipped_pit += 1
                continue
            liq = universe[ticker_market[s.ticker]].get("liq")
            if liq is not None and not liq.is_eligible(s.ticker, wk_date):
                skipped_liq += 1
                continue
            if sector_map is not None and not sector_map.sector_ok(s.ticker, wk_date):
                skipped_sector += 1
                continue

            # prezzo e data di esecuzione
            if execution == "same_close":
                exec_date, exec_px = wk_date, s.entry
            else:
                df_tk = per_ticker[s.ticker]
                after = df_tk.index[df_tk.index > wk_date]
                if len(after) == 0:
                    continue                    # nessuna settimana dopo: salta
                exec_date = after[0]
                nxt = df_tk.loc[exec_date]
                # apertura aggiustata della settimana successiva; se manca,
                # ripiega sulla chiusura di quella settimana
                exec_px = float(nxt.get("adj_open", nxt["adj_close"]))
                if not (np.isfinite(exec_px) and exec_px > 0):
                    exec_px = float(nxt["adj_close"])
                # lo stop resta quello strutturale del segnale; se il gap di
                # apertura è già sotto lo stop, il trade non ha senso: salta
                if exec_px <= s.stop:
                    continue

            eq = pf.equity(prices)
            # FLIP: se si è short su questo titolo, si chiude lo short e si va long
            if s.ticker in pf.positions and pf.positions[s.ticker].side == "short":
                pf.close(s.ticker, exec_date, exec_px, "flip_to_long")
            if rebalance_on_signal and s.ticker not in pf.positions:
                # size target SENZA cap di cassa; se non c'è abbastanza cassa, si
                # libera spazio trimmando una fetta proporzionale delle long
                target = pf.size_fixed_risk(eq, exec_px, s.stop, side="long", cash_cap=False)
                target_cost = target * exec_px * (1 + pf._cost_rate())
                if target_cost > pf.cash:
                    needed = target_cost - pf.cash
                    longs = [(tk, p) for tk, p in pf.positions.items() if p.side == "long"]
                    tot = sum(p.shares * prices.get(tk, p.last_price) for tk, p in longs)
                    if tot > 0:
                        frac = min(1.0, needed / tot)
                        for tk, p in longs:
                            pf.reduce(tk, exec_date, prices.get(tk, p.last_price),
                                      p.shares * frac, "rebalance_trim")
                shares = pf.size_fixed_risk(pf.equity(prices), exec_px, s.stop, side="long")
            else:
                shares = pf.size_fixed_risk(eq, exec_px, s.stop, side="long")
            pf.open(s.ticker, exec_date, exec_px, s.stop, shares)

        # 3d. ingressi SHORT (breakdown di Fase 4), speculari al long ----------
        for s in sig_by_week_short.get(wk_date, []):
            pit = ticker_pit.get(s.ticker)
            if pit is not None and not pit.get(wk_date, False):
                continue
            liq = universe[ticker_market[s.ticker]].get("liq")
            if liq is not None and not liq.is_eligible(s.ticker, wk_date):
                continue
            if sector_map is not None and not sector_map.sector_ok(s.ticker, wk_date):
                continue
            if execution == "same_close":
                exec_date, exec_px = wk_date, s.entry
            else:
                df_tk = per_ticker[s.ticker]
                after = df_tk.index[df_tk.index > wk_date]
                if len(after) == 0:
                    continue
                exec_date = after[0]
                nxt = df_tk.loc[exec_date]
                exec_px = float(nxt.get("adj_open", nxt["adj_close"]))
                if not (np.isfinite(exec_px) and exec_px > 0):
                    exec_px = float(nxt["adj_close"])
                # short: lo stop è SOPRA; se l'apertura gappa già sopra lo stop il
                # trade non ha senso (perdita immediata): salta
                if exec_px >= s.stop:
                    continue
            eq = pf.equity(prices)
            shares = pf.size_fixed_risk(eq, exec_px, s.stop, side="short")
            # FLIP: se si è long su questo titolo, si chiude il long e si va short
            if s.ticker in pf.positions and pf.positions[s.ticker].side == "long":
                pf.close(s.ticker, exec_date, exec_px, "flip_to_short")
            pf.open_short(s.ticker, exec_date, exec_px, s.stop, shares)

        _eq = pf.equity(prices)
        equity_hist.append((wk_date, _eq))
        # ESPOSIZIONE: quanto del capitale è investito (1 - cassa/equity) e quante
        # posizioni. Serve a distinguere il valore della SELEZIONE dall'effetto
        # di "quanto sei sul mercato": in un toro, l'esposizione da sola conta.
        exposure_hist.append(0.0 if _eq <= 0 else 1.0 - pf.cash / _eq)
        npos_hist.append(len(pf.positions))

    equity_curve = pd.Series(dict(equity_hist), name="equity").sort_index()

    # benchmark: buy & hold del benchmark del primo mercato, stesso capitale
    first_bench = bench_data[next(iter(universe))]
    b = first_bench["adj_close"].loc[start:end]
    bench_curve = b / b.iloc[0] * pf_cfg["initial_capital"]

    metrics = compute_metrics(cfg, equity_curve, bench_curve, pf, signal_stats)
    metrics["avg_exposure"] = float(np.mean(exposure_hist)) if exposure_hist else 0.0
    metrics["avg_positions"] = float(np.mean(npos_hist)) if npos_hist else 0.0
    # metriche SHORT (se attivo): quanti segnali, quanti trade, hit rate
    short_trades = [t for t in pf.trades if getattr(t, "side", "long") == "short"]
    metrics["n_short_signals"] = n_short_sig
    metrics["n_short_trades"] = len(short_trades)
    metrics["short_hit_rate"] = (sum(1 for t in short_trades if t.pnl > 0) / len(short_trades)
                                 if short_trades else float("nan"))
    metrics["universe_mode"] = {m: spec.get("mode", "list") for m, spec in universe.items()}
    metrics["signals_skipped_pit"] = skipped_pit
    metrics["signals_skipped_liq"] = skipped_liq
    metrics["signals_skipped_sector"] = skipped_sector
    # Il filtro settoriale è ATTIVO solo se la mappa esiste E gli ETF sono
    # caricati. `sector_map is not None` non basta: una mappa senza ETF non
    # filtra nulla, e il report direbbe il falso.
    metrics["sector_filter_active"] = bool(
        sector_map is not None and sector_map.coverage()["etf_caricati"])
    # Il filtro di volatilità aggiunge selettività, o il turnover lo implica già?
    for spec in universe.values():
        if spec.get("liq") is not None:
            metrics["volatility_check"] = spec["liq"].redundancy_check()
            metrics["liquidity_stats"] = spec["liq"].stats()
            break
    if progress and metrics.get("volatility_check", {}).get("disponibile"):
        vc = metrics["volatility_check"]
        print(f"\n  [volatilità] corr(turnover, ATR%) = {vc['correlazione_turnover_volatilita']}")
        if "interpretazione" in vc:
            print(f"  [volatilità] {vc['interpretazione']} "
                  f"({vc['quota_esclusa_solo_da_volatilita_pct']}% delle settimane)")
    metrics["universe_size"] = len(per_ticker)
    metrics["tickers_skipped_nodata"] = skipped_nodata
    if progress and skipped_nodata:
        print(f"  [universo] {skipped_nodata} ticker saltati (cache vuota o "
              f"storico troppo corto per la MA30)")
    metrics["forced_exits_delisting"] = forced_delist
    if progress and forced_delist:
        print(f"\n  {forced_delist} posizioni chiuse d'ufficio: titolo delistato")
    if progress and (skipped_pit or skipped_liq or skipped_sector):
        print("\n  Segnali scartati dai filtri d'universo:")
        if skipped_pit:
            print(f"    {skipped_pit:4d} · titolo non nell'indice a quella data")
        if skipped_liq:
            print(f"    {skipped_liq:4d} · liquidità sotto soglia")
        if skipped_sector:
            print(f"    {skipped_sector:4d} · settore non in Fase 2 (secondo schermo)")
    # --- Gemello casuale: la strategia batte lo scegliere a caso? ---------
    # Risponde alla domanda sul PICKING (non sul timing di mercato). Attivo
    # solo se richiesto nel config, perché costa n_sims backtest aggiuntivi.
    twin_cfg = cfg.get("random_twin", {})
    if twin_cfg.get("enabled", False):
        from .random_twin import run_twin_distribution, strategy_percentile
        n_sims = twin_cfg.get("n_sims", 100)
        weeks_idx = equity_curve.index
        # match_exposure: cappa il gemello all'esposizione della strategia,
        # settimana per settimana → confronto della SELEZIONE a pari esposizione
        # (toglie il confondente "chi è più investito vince nel toro").
        target_expo = None
        if twin_cfg.get("match_exposure", False):
            target_expo = (pd.Series(exposure_hist, index=weeks)
                           .reindex(weeks_idx).ffill().fillna(0.0).to_numpy())
        if progress:
            print(f"\n  Gemello casuale: {n_sims} sorteggi "
                  f"(market_filter={twin_cfg.get('use_market_filter', True)}, "
                  f"match_exposure={twin_cfg.get('match_exposure', False)})...")
        twin = run_twin_distribution(
            per_ticker, ticker_market, universe, market_stage, weeks_idx, cfg,
            n_sims=n_sims,
            use_market_filter=twin_cfg.get("use_market_filter", True),
            progress=progress, target_expo=target_expo)
        strat_cagr = metrics["strategy"]["cagr"]
        pct = strategy_percentile(strat_cagr, twin["cagrs"])
        twin_summary = {k: v for k, v in twin.items() if k != "cagrs"}
        twin_summary["strategy_cagr"] = strat_cagr
        twin_summary["strategy_percentile"] = pct
        metrics["random_twin"] = twin_summary
        if progress:
            print(f"  → strategia CAGR {strat_cagr*100:+.1f}% · "
                  f"caso mediano {twin['cagr_p50']*100:+.1f}% "
                  f"[{twin['cagr_p05']*100:+.1f}%, {twin['cagr_p95']*100:+.1f}%]")
            print(f"  → la strategia sta al {pct:.0f}° percentile dei sorteggi casuali")
            if pct >= 90:
                print("    il picking aggiunge valore reale ✓")
            elif pct <= 50:
                print("    pescare a caso avrebbe fatto altrettanto o meglio")

    return BacktestResult(equity_curve, bench_curve, pf.trades, all_signals,
                          signal_stats, metrics, per_ticker,
                          market_stage[next(iter(universe))])


# ----------------------------------------------------------------------
def _curve_metrics(curve: pd.Series) -> dict[str, float]:
    if len(curve) < 3:
        return {"cagr": np.nan, "max_dd": np.nan, "sharpe": np.nan}
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    dd = (curve / curve.cummax() - 1).min()
    weekly_ret = curve.pct_change().dropna()
    sharpe = weekly_ret.mean() / weekly_ret.std() * np.sqrt(52) if weekly_ret.std() > 0 else np.nan
    return {"cagr": float(cagr), "max_dd": float(dd), "sharpe": float(sharpe)}


def compute_metrics(cfg: dict, equity: pd.Series, bench: pd.Series,
                    pf: Portfolio, sig_stats: pd.DataFrame) -> dict[str, Any]:
    split = pd.Timestamp(cfg["backtest"]["oos_split"])
    horizons = cfg["backtest"]["horizons_weeks"]
    # `.loc[:split]` e `.loc[split:]` sono ENTRAMBI inclusivi in pandas: se lo
    # split coincide con una data presente, quella settimana finirebbe in
    # entrambi i periodi (double-counting). L'out-of-sample parte STRETTAMENTE
    # dopo lo split, così i due insiemi sono disgiunti per costruzione.
    after_split = equity.index[equity.index > split]
    oos_start = after_split[0] if len(after_split) else split
    out: dict[str, Any] = {"strategy": _curve_metrics(equity),
                           "benchmark": _curve_metrics(bench)}
    out["strategy_is"] = _curve_metrics(equity.loc[:split])
    out["strategy_oos"] = _curve_metrics(equity.loc[oos_start:])
    out["benchmark_is"] = _curve_metrics(bench.loc[:split])
    out["benchmark_oos"] = _curve_metrics(bench.loc[oos_start:])

    wins = [t for t in pf.trades if t.pnl > 0]
    out["n_trades"] = len(pf.trades)
    out["hit_rate_trades"] = len(wins) / len(pf.trades) if pf.trades else np.nan
    out["avg_win"] = float(np.mean([t.ret for t in wins])) if wins else np.nan
    losses = [t for t in pf.trades if t.pnl <= 0]
    out["avg_loss"] = float(np.mean([t.ret for t in losses])) if losses else np.nan

    out["n_signals"] = int(len(sig_stats))
    sig_summary = {}
    if len(sig_stats):
        for seg, mask in [("all", pd.Series(True, index=sig_stats.index)),
                          ("is", sig_stats["date"] < split),
                          ("oos", sig_stats["date"] >= split)]:
            seg_df = sig_stats[mask]
            row = {"n": int(len(seg_df))}
            for h in horizons:
                exc = seg_df[f"exc_{h}w"].dropna()
                row[f"hit_{h}w"] = float((exc > 0).mean()) if len(exc) else np.nan
                row[f"avg_exc_{h}w"] = float(exc.mean()) if len(exc) else np.nan
            sig_summary[seg] = row
    out["signal_summary"] = sig_summary
    return out
