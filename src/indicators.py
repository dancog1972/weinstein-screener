"""Indicatori del metodo: funzioni pure su DataFrame settimanali.

Nessuna funzione guarda "avanti": ogni valore alla settimana t usa solo dati
fino a t compreso (il volume di confronto usa .shift(1): solo settimane
PRECEDENTI). Garanzia anti look-ahead a livello di indicatori.
"""
from __future__ import annotations

import pandas as pd


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Ricampiona barre giornaliere in settimanali (chiusura del venerdì)."""
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "adj_close": "last", "volume": "sum"}
    return daily.resample("W-FRI").agg(agg).dropna(subset=["close"])


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def ma_slope(ma: pd.Series, lookback: int) -> pd.Series:
    """Variazione percentuale della MA sulle ultime `lookback` settimane."""
    return ma.pct_change(periods=lookback)


def mansfield_rs(close: pd.Series, bench_close: pd.Series, n: int = 52) -> pd.Series:
    """Mansfield RS (52 settimane, come in Weinstein):
    RS = close/benchmark; Mansfield = (RS / SMA_n(RS) - 1) * 100."""
    bench = bench_close.reindex(close.index).ffill()
    rs = close / bench
    return (rs / sma(rs, n) - 1.0) * 100.0


def volume_ratio(volume: pd.Series, n: int = 4) -> pd.Series:
    """Volume della settimana / media delle n settimane PRECEDENTI."""
    prior_avg = volume.shift(1).rolling(n, min_periods=n).mean()
    return volume / prior_avg


def confirmed_pivot_lows(weekly: pd.DataFrame, k: int = 2) -> pd.Series:
    """Minimi di pullback CONFERMATI, senza look-ahead.

    Un pivot low alla settimana p esiste se low[p] è il minimo delle k
    settimane precedenti e delle k successive — ma lo si SA solo alla
    settimana p+k. La serie restituita vale, alla settimana t, il prezzo
    del pivot confermato proprio in t (low di t-k), altrimenti NaN.
    È la base del trailing stop 'da trader' di Weinstein: a ogni minimo
    di correzione più alto, lo stop sale sotto quel minimo.

    Sui PAREGGI: la condizione è che nessuna barra della finestra scenda
    SOTTO il pivot, non che tutte stiano strettamente sopra. Un doppio
    minimo (low[p] == low[p+1]) è una formazione reale e frequente — il
    double bottom di Weinstein — e deve produrre un pivot. Per evitare di
    contarlo due volte, in caso di pareggio si tiene solo la PRIMA barra
    (la più a sinistra) come pivot.
    """
    low = weekly["adj_low"] if "adj_low" in weekly else weekly["low"]
    n = len(low)
    out = pd.Series(float("nan"), index=weekly.index)
    vals = low.to_numpy()
    for t in range(2 * k, n):
        p = t - k
        window = vals[p - k: p + k + 1]
        if vals[p] != window.min():
            continue
        # nessuna barra scende sotto il pivot (i pareggi sono ammessi)
        # e il pivot è il PRIMO a toccare quel minimo nella finestra
        left = vals[p - k: p]
        if len(left) and left.min() <= vals[p]:
            continue          # un pareggio a sinistra: quello è il pivot, non questo
        out.iloc[t] = vals[p]
    return out


def last_confirmed_pivot_before(weekly: pd.DataFrame, when: pd.Timestamp,
                                k: int = 2, max_lookback: int = 30) -> float | None:
    """L'ultimo minimo relativo CONFERMATO prima della settimana `when`.

    È l'ancora dello stop secondo Weinstein: non il fondo della base (che su
    una base lunga è lontanissimo e irrilevante), ma l'ultima oscillazione
    strutturale che precede il breakout — il primo supporto vero sotto il
    prezzo.

    Anti look-ahead: un pivot alla settimana p è confermato solo in p+k, e
    qui accettiamo solo pivot la cui CONFERMA cade prima di `when`. Quindi
    al momento del segnale quel minimo era già noto.

    Ritorna None se nessun pivot è confermato entro `max_lookback` settimane:
    in quel caso non esiste un supporto strutturale utilizzabile e chi chiama
    deve ripiegare su un altro criterio.
    """
    piv = confirmed_pivot_lows(weekly, k)
    idx = weekly.index
    if when not in idx:
        return None
    pos = idx.get_loc(when)
    start = max(0, pos - max_lookback)
    window = piv.iloc[start:pos]          # confermati STRETTAMENTE prima di `when`
    valid = window.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def confirmed_pivot_highs(weekly: pd.DataFrame, k: int = 2) -> pd.Series:
    """Simmetrico di confirmed_pivot_lows per lo SHORT: massimi di rimbalzo
    confermati. Un pivot high a p esiste se high[p] è il massimo delle k
    settimane precedenti e successive, noto solo in p+k. Ancora del trailing
    stop short: a ogni massimo di rimbalzo più basso, lo stop scende sopra."""
    high = weekly["adj_high"] if "adj_high" in weekly else weekly["high"]
    n = len(high)
    out = pd.Series(float("nan"), index=weekly.index)
    vals = high.to_numpy()
    for t in range(2 * k, n):
        p = t - k
        window = vals[p - k: p + k + 1]
        if vals[p] != window.max():
            continue
        left = vals[p - k: p]
        if len(left) and left.max() >= vals[p]:
            continue
        out.iloc[t] = vals[p]
    return out


def last_confirmed_pivot_high_before(weekly: pd.DataFrame, when: pd.Timestamp,
                                     k: int = 2, max_lookback: int = 30) -> float | None:
    """L'ultimo massimo relativo CONFERMATO prima di `when`. Ancora dello stop
    SHORT: lo stop va sopra l'ultimo rimbalzo strutturale prima del breakdown.
    Stesso anti-look-ahead di last_confirmed_pivot_before."""
    piv = confirmed_pivot_highs(weekly, k)
    idx = weekly.index
    if when not in idx:
        return None
    pos = idx.get_loc(when)
    start = max(0, pos - max_lookback)
    window = piv.iloc[start:pos]
    valid = window.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def atr(weekly: pd.DataFrame, n: int = 10) -> pd.Series:
    """ATR su prezzi AGGIUSTATI quando disponibili: un ATR calcolato su high/low
    grezzi e poi diviso per adj_close (come fa il filtro di volatilità) darebbe
    un ATR% completamente sbagliato sui titoli con split."""
    hi = weekly["adj_high"] if "adj_high" in weekly else weekly["high"]
    lo = weekly["adj_low"] if "adj_low" in weekly else weekly["low"]
    cl = weekly["adj_close"] if "adj_close" in weekly else weekly["close"]
    hl = hi - lo
    hc = (hi - cl.shift(1)).abs()
    lc = (lo - cl.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def enrich_weekly(weekly: pd.DataFrame, bench_weekly: pd.DataFrame | None,
                  ma_weeks: int, slope_lookback: int) -> pd.DataFrame:
    """Aggiunge al DataFrame settimanale tutte le colonne usate dal motore."""
    out = weekly.copy()

    # High/Low AGGIUSTATI. Cruciale: `adj_close` è riscalato all'indietro per
    # split e dividendi, mentre `high`/`low` grezzi NON lo sono. Su AAPL (split
    # 7:1 nel 2014) il close grezzo pre-split vale ~$650 e l'adj_close ~$93:
    # confrontare `adj_close > resistance` dove la resistenza viene dagli high
    # grezzi mette a confronto scale diverse, e il breakout non scatta mai —
    # o scatta a caso. Applichiamo lo stesso fattore di aggiustamento.
    factor = out["adj_close"] / out["close"].replace(0, pd.NA)
    factor = factor.ffill().bfill().fillna(1.0)
    out["adj_high"] = out["high"] * factor
    out["adj_low"] = out["low"] * factor
    out["adj_open"] = out["open"] * factor      # per l'esecuzione a t+1

    out["ma"] = sma(out["adj_close"], ma_weeks)
    out["ma_slope"] = ma_slope(out["ma"], slope_lookback)
    out["vol_ratio"] = volume_ratio(out["volume"])
    if bench_weekly is not None:
        out["mansfield"] = mansfield_rs(out["adj_close"], bench_weekly["adj_close"])
    else:
        out["mansfield"] = 0.0
    out["atr"] = atr(out)
    return out
