"""Rilevamento del segnale di acquisto: breakout di Fase 2 alla Weinstein.

Condizioni (tutte parametriche da config, valutate alla chiusura settimanale t):
  1. La settimana PRECEDENTE era in Fase 1 con base di durata >= min_base_weeks
     e profondità <= max_base_depth (qualità della base).
  2. Chiusura t > resistenza della base e > MA30, con MA non in discesa.
  3. Volume t >= volume_ratio_min x media delle 4 settimane precedenti.
  4. Mansfield RS >= mansfield_min e in crescita sulle ultime N settimane.
  5. (market_filter) il benchmark del titolo è in Fase 2 alla settimana t.
Il segnale è calcolato SOLO con dati fino a t: eseguibile all'apertura successiva.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import (last_confirmed_pivot_before,
                         last_confirmed_pivot_high_before)
from .stages import base_features, classify_stages, top_features


@dataclass
class Signal:
    ticker: str
    date: pd.Timestamp          # venerdì della settimana di breakout
    entry: float                # chiusura del breakout (l'esecuzione aggiunge slippage)
    stop: float                 # stop iniziale
    resistance: float
    base_len: int
    base_depth: float
    vol_ratio: float
    mansfield: float
    extras: dict = field(default_factory=dict)


def prepare_ticker(enriched: pd.DataFrame, flat_slope: float) -> pd.DataFrame:
    """Aggiunge fase e caratteristiche BASE (long) e TOP (short) al settimanale
    arricchito. Così un solo df serve sia i segnali long sia quelli short."""
    stages = classify_stages(enriched, flat_slope)
    out = enriched.join(base_features(enriched, stages))
    out = out.join(top_features(enriched, stages))    # feature del top per lo short
    out["stage"] = stages
    return out


def prepare_ticker_short(enriched: pd.DataFrame, flat_slope: float) -> pd.DataFrame:
    """Simmetrico per lo SHORT: aggiunge fase e caratteristiche del TOP."""
    stages = classify_stages(enriched, flat_slope)
    feats = top_features(enriched, stages)
    out = enriched.join(feats)
    out["stage"] = stages
    return out


def initial_stop(df: pd.DataFrame, ts: pd.Timestamp, entry: float,
                 resistance: float, exit_cfg: dict) -> float:
    """Stop iniziale del breakout long. UNICA definizione, CONDIVISA tra
    detect_signals (motore validato) e lo screener, così i due non possono
    divergere sul livello di stop.

    pivot: sotto l'ultimo minimo RELATIVO confermato prima del breakout — non il
    fondo della base, che su una base lunga è lontanissimo e perde sempre il
    confronto col buffer. Se non c'è un pivot valido sotto l'entrata, si ripiega
    sotto la resistenza della base (stop_buffer, dichiarato). Modalità 'buffer':
    max(minimo della base, resistenza − buffer)."""
    if exit_cfg.get("stop_mode", "buffer") == "pivot":
        piv = last_confirmed_pivot_before(df, ts, exit_cfg.get("pivot_k", 2))
        if piv is not None and piv < entry:
            return float(piv) * (1.0 - exit_cfg.get("pivot_buffer", 0.01))
        return resistance * (1.0 - exit_cfg["stop_buffer"])
    low_prev = float(df["base_low"].shift(1).loc[ts])
    return max(low_prev, resistance * (1.0 - exit_cfg["stop_buffer"]))


def detect_signals(ticker: str, df: pd.DataFrame, market_stage: pd.Series | None,
                   sig_cfg: dict, exit_cfg: dict, flat_slope: float = 0.005) -> list[Signal]:
    """df: output di prepare_ticker. market_stage: fase settimanale del benchmark.
    flat_slope: la MA al breakout può essere piatta (|pendenza| < flat_slope) o in
    salita, ma non significativamente in discesa — come nel libro."""
    res_prev = df["resistance"].shift(1)
    len_prev = df["base_len"].shift(1)
    depth_prev = df["base_depth"].shift(1)
    mans_rising = df["mansfield"].diff(sig_cfg["mansfield_rising_weeks"]) > 0

    cond = (
        (len_prev >= sig_cfg["min_base_weeks"])
        & (depth_prev <= sig_cfg["max_base_depth"])
        & (df["adj_close"] > res_prev)
        & (df["adj_close"] > df["ma"])
        & (df["ma_slope"] > -flat_slope)                # MA piatta o in salita
        & (df["vol_ratio"] >= sig_cfg["volume_ratio_min"])
        & (df["mansfield"] >= sig_cfg["mansfield_min"])
    )
    # il filtro sulla DIREZIONE (Mansfield in salita) è disattivabile: serve
    # per l'analisi dei quadranti, che deve vedere anche i segnali in discesa
    if sig_cfg.get("require_mansfield_rising", True):
        cond = cond & mans_rising
    # Requisito A (Weinstein): la base deve essere ACCUMULAZIONE DOPO UN DECLINO.
    # Se richiesto ma la colonna manca, si ALZA: significa un settimanale stale
    # (cache scritta prima che after_decline esistesse). Ignorarlo in silenzio
    # disattivava il filtro senza accorgersene — bug reale. La wkey ora versiona
    # lo schema (WEEKLY_SCHEMA_VERSION), quindi la cache si rigenera da sola.
    if sig_cfg.get("require_decline", True):
        if "after_decline" not in df.columns:
            raise ValueError(
                "require_decline=true ma manca la colonna 'after_decline': "
                "settimanale stale. Rigenera la cache (cancella data/*/_weekly)."
            )
        cond = cond & df["after_decline"].shift(1).fillna(False)
    # Contrazione del VOLUME nella base (Weinstein: nella base il volume si
    # prosciuga, poi esplode al breakout). base_vol_ratio = volume medio nella
    # base / volume medio prima della base; richiediamo che alla fine della base
    # (shift(1)) fosse <= soglia. NaN (riferimento mancante, inizio serie) passa:
    # non rifiutiamo per dato assente. Disattivato se la soglia non è in config.
    # Volume nella base a DUE livelli (idea Weinstein/Wyckoff): passa chi ha il
    # DRY-UP (contrazione nella prima parte della base) OPPURE chi mostra ACCUMULO
    # (OBV di base in salita: più volume sulle settimane di rialzo). Un titolo che
    # non si è asciugato ma è in accumulo resta un candidato valido. Se base_obv_min
    # non è impostato, vale il solo dry-up.
    bvr_max = sig_cfg.get("base_volume_max_ratio")
    obv_min = sig_cfg.get("base_obv_min")
    if bvr_max is not None and "base_vol_ratio" in df.columns:
        is_dry = ~(df["base_vol_ratio"].shift(1) > bvr_max)
        if obv_min is not None and "base_obv" in df.columns:
            is_accum = df["base_obv"].shift(1) >= obv_min
            cond = cond & (is_dry | is_accum)
        else:
            cond = cond & is_dry
    # Breakout da NOTIZIA: una chiusura troppo SOPRA la resistenza ha scavalcato
    # la base con un gap (spike da news), non l'ha rotta in modo ordinato — e
    # spesso ritraccia. Weinstein/uno scanner lo segnalerebbero al giudizio umano;
    # in automatico lo si ignora. Firma: (close/resistenza - 1) > soglia.
    max_stretch = sig_cfg.get("max_breakout_stretch")
    if max_stretch is not None:
        cond = cond & ~((df["adj_close"] / res_prev - 1.0) > max_stretch)
    if sig_cfg.get("market_filter", True) and market_stage is not None:
        ms = market_stage.reindex(df.index).ffill()
        cond = cond & (ms == 2)

    out: list[Signal] = []
    for ts in df.index[cond.fillna(False)]:
        row = df.loc[ts]
        resistance = float(res_prev.loc[ts])
        entry = float(row["adj_close"])

        # Seconda rete contro i dati corrotti sfuggiti alla sanificazione:
        # un entry non finito o <= 0, un volume ratio infinito (media
        # precedente zero), un Mansfield fuori scala (>500% = benchmark o
        # prezzo corrotto) non sono breakout, sono errori. Scartarli qui
        # evita che una manciata di righe con rendimenti a quattro-cifre
        # avveleni ogni media aggregata.
        vr = float(row["vol_ratio"])
        mans = float(row["mansfield"])
        if not (np.isfinite(entry) and entry > 0):
            continue
        if not np.isfinite(vr) or vr > 100:
            continue
        if not np.isfinite(mans) or abs(mans) > 500:
            continue
        if not (np.isfinite(resistance) and resistance > 0):
            continue

        # stop iniziale: UNA sola definizione, condivisa con lo screener (vedi
        # initial_stop) così motore e scanner non possono divergere sul livello.
        stop = initial_stop(df, ts, entry, resistance, exit_cfg)

        # uno stop sopra o pari all'ingresso non ha senso: scarta il segnale
        if stop >= entry:
            continue
        # filtro opzionale sul rischio: un breakout col supporto lontanissimo
        # ha rapporto rendimento/rischio sfavorevole (Weinstein li evita)
        max_risk = sig_cfg.get("max_risk_pct")
        if max_risk and (entry - stop) / entry > max_risk:
            continue
        out.append(Signal(
            ticker=ticker, date=ts, entry=entry, stop=stop,
            resistance=resistance, base_len=int(len_prev.loc[ts]),
            base_depth=float(depth_prev.loc[ts]),
            vol_ratio=float(row["vol_ratio"]), mansfield=float(row["mansfield"]),
        ))
    return out


def detect_signals_short(ticker, df, market_stage, sig_cfg, exit_cfg, flat_slope=0.005):
    """Simmetrico ribassista di detect_signals. Il breakdown scatta quando il
    prezzo rompe SOTTO il supporto di un top (Fase 3), con MA30 in discesa,
    Mansfield negativo e calante, volume alto. Lo stop va SOPRA l'ingresso,
    sull'ultimo massimo confermato. Per lo SHORT: si guadagna quando scende."""
    height_prev = df["top_height_est"].shift(1)
    mans_falling = df["mansfield"].diff(sig_cfg["mansfield_rising_weeks"]) < 0

    cond = (
        (df["support_est"].notna())
        & (height_prev <= sig_cfg["max_base_depth"])
        & (df["adj_close"] < df["support_est"])
        & (df["adj_close"] < df["ma"])
        & (df["ma_slope"] < flat_slope)
        & (df["vol_ratio"] >= sig_cfg["volume_ratio_min"])
        & (df["mansfield"] <= -sig_cfg["mansfield_min"])
        & mans_falling
    )
    if sig_cfg.get("require_decline", True) and "after_advance_est" in df.columns:
        cond = cond & df["after_advance_est"].fillna(False)
    # Volume nel TOP a due livelli, specchio del long: passa chi ha DRY-UP nel top
    # OPPURE DISTRIBUZIONE (OBV negativo: più volume sui ribassi). top_vol_ratio e
    # top_obv sono già congelati (pre-breakdown), quindi niente shift.
    bvr_max = sig_cfg.get("base_volume_max_ratio")
    obv_min = sig_cfg.get("base_obv_min")
    if bvr_max is not None and "top_vol_ratio" in df.columns:
        is_dry = ~(df["top_vol_ratio"] > bvr_max)
        if obv_min is not None and "top_obv" in df.columns:
            is_distrib = df["top_obv"] <= -obv_min
            cond = cond & (is_dry | is_distrib)
        else:
            cond = cond & is_dry
    # anti-spike speculare: un crollo troppo SOTTO il supporto è panico/notizia
    max_stretch = sig_cfg.get("max_breakout_stretch")
    if max_stretch is not None:
        cond = cond & ~((df["support_est"] - df["adj_close"]) / df["support_est"] > max_stretch)
    if sig_cfg.get("market_filter", True) and market_stage is not None:
        ms = market_stage.reindex(df.index).ffill()
        confirm = int(sig_cfg.get("bear_confirm_weeks", 0) or 0)
        if confirm > 1:
            # ORSO CONFERMATO: si shorta solo se il mercato è in Fase 4 da almeno
            # `confirm` settimane consecutive — non sul primo tuffo (spesso un
            # falso allarme dentro il toro, poi bear-rally che fa scattare lo stop).
            in_bear = (ms == 4).astype(int)
            confirmed = in_bear.rolling(confirm, min_periods=confirm).sum() == confirm
            cond = cond & confirmed.fillna(False)
        else:
            cond = cond & (ms == 4)

    out = []
    pivot_k = exit_cfg.get("pivot_k", 2)
    for ts in df.index[cond.fillna(False)]:
        row = df.loc[ts]
        support = float(df["support_est"].loc[ts])
        entry = float(row["adj_close"])
        vr = float(row["vol_ratio"])
        mans = float(row["mansfield"])
        if not (np.isfinite(entry) and entry > 0):
            continue
        if not np.isfinite(vr) or vr > 100:
            continue
        if not np.isfinite(mans) or abs(mans) > 500:
            continue
        if not (np.isfinite(support) and support > 0):
            continue

        piv = last_confirmed_pivot_high_before(df, ts, pivot_k)
        if piv is not None and piv > entry:
            stop = float(piv) * (1.0 + exit_cfg.get("pivot_buffer", 0.01))
        else:
            stop = support * (1.0 + exit_cfg["stop_buffer"])

        if stop <= entry:
            continue
        max_risk = sig_cfg.get("max_risk_pct")
        if max_risk and (stop - entry) / entry > max_risk:
            continue
        out.append(Signal(
            ticker=ticker, date=ts, entry=entry, stop=stop,
            resistance=support, base_len=int(df["top_len"].loc[ts]),
            base_depth=float(df["top_height_est"].loc[ts]),
            vol_ratio=float(row["vol_ratio"]), mansfield=float(row["mansfield"]),
        ))
    return out




def forward_returns(sig: Signal, df: pd.DataFrame, bench: pd.DataFrame,
                    horizons: list[int]) -> dict[str, float]:
    """Rendimenti forward del segnale e in eccesso sul benchmark, per orizzonte.
    Usati per valutare la QUALITÀ dei segnali indipendentemente dal portafoglio."""
    out: dict[str, float] = {}
    idx = df.index.get_loc(sig.date)
    b_idx = bench.index.get_loc(bench.index.asof(sig.date))
    for h in horizons:
        if idx + h < len(df) and b_idx + h < len(bench):
            r = df["adj_close"].iloc[idx + h] / sig.entry - 1.0
            rb = bench["adj_close"].iloc[b_idx + h] / bench["adj_close"].iloc[b_idx] - 1.0
            out[f"fwd_{h}w"] = float(r)
            out[f"exc_{h}w"] = float(r - rb)
        else:
            out[f"fwd_{h}w"] = np.nan
            out[f"exc_{h}w"] = np.nan
    return out
