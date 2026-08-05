"""Portafoglio gemello casuale: il metro per misurare il valore del picking.

La domanda vera non è "batto l'S&P?" (timing di mercato) ma "la mia SELEZIONE
batte lo scegliere a caso tra i titoli idonei?". Il gemello risponde: stesso
universo, stesso capitale, stessi costi, stesso market filter, stessi vincoli
(nessun tetto di posizioni) — differisce SOLO nella regola di selezione. Invece
di aspettare un segnale Weinstein, quando c'è cash pesca un titolo a caso tra
quelli idonei per liquidità in quella settimana.

Ripetuto su molti semi → distribuzione: la strategia sta al 50° percentile
(uguale al caso) o al 95° (selezione genuinamente buona)?

Il gemello NON usa stop: entra a caso ed esce solo quando il titolo lascia
l'universo. Isola il valore della scelta di COSA comprare.

Prestazioni: il pannello (prezzi + idoneità) si pre-calcola UNA volta in array
NumPy e si condivide fra tutti i sorteggi. La versione ingenua richiamava
is_eligible per ogni titolo, ogni settimana, ogni sorteggio (~1 miliardo di
operazioni su 100 sorteggi = ore). Con il pannello: ~16 milioni = minuti.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _precompute_panel(
    per_ticker: dict[str, pd.DataFrame],
    ticker_market: dict[str, str],
    universe: dict,
    weeks: pd.DatetimeIndex,
) -> dict:
    """Matrici (settimane × titoli) costruite UNA volta e riusate da ogni
    sorteggio: prezzi e 'tradabile' (idoneo per liquidità E con prezzo valido)."""
    tickers = list(per_ticker.keys())
    price_cols, elig_cols = [], []
    ma_cols, slope_cols, atr_cols, piv_cols = [], [], [], []
    for tk in tickers:
        d = per_ticker[tk]
        px = d["adj_close"].reindex(weeks).to_numpy(dtype=float)
        price_cols.append(px)
        liq = universe[ticker_market[tk]].get("liq")
        if liq is not None and tk in getattr(liq, "_eligible", {}):
            e = liq._eligible[tk].reindex(weeks).fillna(False).to_numpy(dtype=bool)
        else:
            e = ~np.isnan(px)
        elig_cols.append(e)
        # colonne per la gestione delle uscite "alla strategia" (manage_exits):
        # MA30, pendenza MA, ATR, e l'ultimo minimo-pivot confermato (ffill).
        ma_cols.append(d["ma"].reindex(weeks).to_numpy(dtype=float)
                       if "ma" in d else np.full(len(weeks), np.nan))
        slope_cols.append(d["ma_slope"].reindex(weeks).to_numpy(dtype=float)
                          if "ma_slope" in d else np.full(len(weeks), np.nan))
        atr_cols.append(d["atr"].reindex(weeks).to_numpy(dtype=float)
                        if "atr" in d else np.full(len(weeks), np.nan))
        if "swing_low" in d:
            piv = d["swing_low"].reindex(d.index).ffill().reindex(weeks).ffill()
            piv_cols.append(piv.to_numpy(dtype=float))
        else:
            piv_cols.append(np.full(len(weeks), np.nan))

    prices = np.column_stack(price_cols)          # (W, T)
    elig = np.column_stack(elig_cols)             # (W, T)
    tradable = elig & (~np.isnan(prices))
    # ultima settimana con prezzo valido per ogni titolo: oltre quella il titolo
    # è DELISTATO (fine dati), non un buco mid-serie. Serve per valutare a 0 solo
    # i veri delisting, non le settimane mancanti in mezzo.
    last_valid = np.full(len(tickers), -1, dtype=int)
    for j, px in enumerate(price_cols):
        ok = np.where(~np.isnan(px))[0]
        last_valid[j] = ok[-1] if len(ok) else -1
    return {"tickers": tickers, "prices": prices, "tradable": tradable,
            "last_valid": last_valid,
            "ma": np.column_stack(ma_cols), "ma_slope": np.column_stack(slope_cols),
            "atr": np.column_stack(atr_cols), "pivot": np.column_stack(piv_cols)}


def run_random_twin(
    per_ticker: dict[str, pd.DataFrame],
    ticker_market: dict[str, str],
    universe: dict,
    market_stage: dict,
    weeks: pd.DatetimeIndex,
    cfg: dict,
    seed: int,
    use_market_filter: bool = True,
    panel: dict | None = None,
    target_expo: np.ndarray | None = None,
) -> pd.Series:
    """Una singola simulazione casuale → curva equity. Con `panel` pre-calcolato
    riusa le matrici invece di riscorrere i titoli."""
    rng = np.random.default_rng(seed)
    pf_cfg = cfg["portfolio"]
    initial = float(pf_cfg["initial_capital"])
    risk = float(pf_cfg["risk_per_trade"])
    slip = float(pf_cfg["slippage_bps"])
    comm = float(pf_cfg["commission_bps"])

    if panel is None:
        panel = _precompute_panel(per_ticker, ticker_market, universe, weeks)
    prices = panel["prices"]
    tradable = panel["tradable"]
    last_valid = panel["last_valid"]
    # stesso metro della strategia: un delisting vale 0 (fallimento) o l'ultimo
    # prezzo noto. Deve essere identico per gemello e strategia.
    delist_zero = cfg.get("exits", {}).get("delisting_value", "last_price") == "zero"

    market = next(iter(universe))
    mkt = market_stage[market].reindex(weeks, method="ffill").to_numpy()

    # manage_exits: dà al gemello le STESSE uscite della strategia (stop iniziale
    # su pivot, trailing MA30 con buffer ATR, rottura MA30). Così il gemello
    # differisce dalla strategia SOLO nella selezione → test perfettamente pulito.
    manage = cfg.get("random_twin", {}).get("manage_exits", False)
    ex = cfg.get("exits", {})
    piv_buf = float(ex.get("pivot_buffer", 0.02))
    k_atr = ex.get("ma_trail_atr_mult")
    ma_buf = float(ex.get("ma_trail_buffer", 0.05))
    ma_brk = bool(ex.get("ma_breakdown", True))
    max_pos_pct = float(pf_cfg.get("max_position_pct", 1.0))
    ma_p, slope_p, atr_p, piv_p = panel["ma"], panel["ma_slope"], panel["atr"], panel["pivot"]

    cash = initial
    # posizioni: idx -> [azioni, ultimo prezzo, stop]. Il delisting si liquida
    # all'ultimo prezzo valido (o 0 se delist_zero), MAI al prezzo d'entrata.
    holdings: dict[int, list] = {}
    equity_hist = np.empty(len(weeks))
    expo_hist = np.empty(len(weeks))      # frazione investita, per confronto
    npos_hist = np.empty(len(weeks))

    for wi in range(len(weeks)):
        px_row = prices[wi]
        trad_row = tradable[wi]

        # valuta e gestisci le posizioni
        pos_value = 0.0
        for ti in list(holdings):
            sh, last_px, stop = holdings[ti]
            p = px_row[ti]
            # 1) uscita dall'universo (delisting / non più liquido)
            if np.isnan(p) or not trad_row[ti]:
                if not np.isnan(p):
                    exit_px = p                       # illiquido: esce a mercato
                elif delist_zero and wi > last_valid[ti]:
                    exit_px = 0.0                     # delisting vero → 0
                else:
                    exit_px = last_px                 # ultimo prezzo (o buco mid-serie)
                cash += sh * exit_px * (1 - slip / 10_000)
                del holdings[ti]
                continue
            # 2) uscite "alla strategia" (solo con manage_exits)
            if manage:
                ma = ma_p[wi, ti]
                atr = atr_p[wi, ti]
                if np.isfinite(ma):
                    new_stop = ((ma - k_atr * atr) if (k_atr is not None and np.isfinite(atr))
                                else ma * (1.0 - ma_buf))
                    if new_stop > stop:               # trailing MA30, solo in salita
                        stop = new_stop
                if p <= stop:                         # stop sulla chiusura
                    cash += sh * stop * (1 - slip / 10_000)
                    del holdings[ti]
                    continue
                if (ma_brk and np.isfinite(ma) and p < ma
                        and np.isfinite(slope_p[wi, ti]) and slope_p[wi, ti] < 0):
                    cash += sh * p * (1 - slip / 10_000)   # rottura MA30 (Fase 4)
                    del holdings[ti]
                    continue
            pos_value += sh * p
            holdings[ti] = [sh, p, stop]

        equity = cash + pos_value
        equity_hist[wi] = equity
        expo_hist[wi] = 0.0 if equity <= 0 else 1.0 - cash / equity
        npos_hist[wi] = len(holdings)

        if use_market_filter and not (mkt[wi] == 2):
            continue

        # cap di ESPOSIZIONE (opzionale): il gemello si ferma alla stessa frazione
        # investita che aveva la strategia quella settimana, così il confronto
        # misura la SELEZIONE a pari esposizione, non "chi è più sul mercato".
        tgt = 1.0 if target_expo is None else float(target_expo[wi])
        if equity > 0 and (1.0 - cash / equity) >= tgt:
            continue

        cand = [ti for ti in np.where(trad_row)[0] if ti not in holdings]
        rng.shuffle(cand)
        for ti in cand:
            p = px_row[ti]
            if np.isnan(p) or p <= 0:
                continue
            if manage:
                # stop iniziale come la strategia: sotto l'ultimo pivot confermato
                piv = piv_p[wi, ti]
                stop0 = piv * (1.0 - piv_buf) if (np.isfinite(piv) and piv < p) else p * 0.75
                if stop0 >= p:
                    continue
                shares = (equity * risk) / (p - stop0)
                shares = min(shares, (equity * max_pos_pct) / p)
            else:
                # gemello classico: nessuno stop reale, sizing con stop fittizio -25%
                stop0 = p * 0.75
                shares = (equity * risk) / (p * 0.25)
            cost = shares * p * (1 + slip / 10_000 + comm / 10_000)
            if shares <= 0:
                continue
            if cost > cash:
                if manage:
                    continue          # size variabile: prova un altro candidato
                break                 # size fissa: se non basta qui, non basta per nessuno
            cash -= cost
            holdings[ti] = [shares, p, stop0]
            if equity > 0 and (1.0 - cash / equity) >= tgt:
                break                             # raggiunta l'esposizione target

    out = pd.Series(equity_hist, index=weeks, name=f"twin_{seed}")
    out.attrs["avg_exposure"] = float(expo_hist.mean())
    out.attrs["avg_positions"] = float(npos_hist.mean())
    return out


def run_twin_distribution(
    per_ticker: dict[str, pd.DataFrame],
    ticker_market: dict[str, str],
    universe: dict,
    market_stage: dict,
    weeks: pd.DatetimeIndex,
    cfg: dict,
    n_sims: int = 100,
    use_market_filter: bool = True,
    progress: bool = False,
    target_expo: np.ndarray | None = None,
) -> dict:
    """Molti sorteggi → distribuzione. Il pannello si pre-calcola una volta e
    si condivide: è ciò che rende 100 simulazioni minuti invece di ore.
    `target_expo`: se dato, cappa l'esposizione del gemello a quella della
    strategia (confronto della SELEZIONE a pari esposizione)."""
    from .backtest import _curve_metrics

    panel = _precompute_panel(per_ticker, ticker_market, universe, weeks)
    if progress:
        print(f"    pannello pronto: {panel['prices'].shape[0]} settimane × "
              f"{panel['prices'].shape[1]} titoli", flush=True)
    import time as _t
    _t0 = _t.time()
    cagrs, dds, expos, nposs = [], [], [], []
    for i in range(n_sims):
        curve = run_random_twin(per_ticker, ticker_market, universe,
                                market_stage, weeks, cfg, seed=i,
                                use_market_filter=use_market_filter, panel=panel,
                                target_expo=target_expo)
        m = _curve_metrics(curve)
        cagrs.append(m["cagr"])
        dds.append(m["max_dd"])
        expos.append(curve.attrs.get("avg_exposure", np.nan))
        nposs.append(curve.attrs.get("avg_positions", np.nan))
        if progress and (i + 1) % 10 == 0:
            _el = _t.time() - _t0
            _eta = _el / (i + 1) * (n_sims - i - 1)
            print(f"    twin {i+1}/{n_sims} · ~{_eta/60:.1f} min rimanenti", flush=True)

    cagrs = np.array(cagrs)
    return {
        "n_sims": n_sims,
        "cagr_mean": float(cagrs.mean()),
        "cagr_std": float(cagrs.std()),
        "cagr_p05": float(np.percentile(cagrs, 5)),
        "cagr_p50": float(np.percentile(cagrs, 50)),
        "cagr_p95": float(np.percentile(cagrs, 95)),
        "dd_mean": float(np.mean(dds)),
        "avg_exposure": float(np.nanmean(expos)) if expos else float("nan"),
        "avg_positions": float(np.nanmean(nposs)) if nposs else float("nan"),
        "cagrs": cagrs.tolist(),
        "use_market_filter": use_market_filter,
    }


def strategy_percentile(strategy_cagr: float, twin_cagrs: list[float]) -> float:
    """A che percentile della distribuzione casuale cade la strategia?
    95 = batte il 95% dei sorteggi (picking eccellente). 50 = uguale al caso."""
    arr = np.array(twin_cagrs)
    return float((arr < strategy_cagr).mean() * 100)
