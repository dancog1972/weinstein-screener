"""Classificazione delle 4 fasi di Weinstein — funzione pura, senza I/O.

Regole (settimanali, sul prezzo rettificato):
  Fase 2 (avanzata): chiusura > MA30 e MA in salita (pendenza > +flat_slope)
  Fase 4 (declino):  chiusura < MA30 e MA in discesa (pendenza < -flat_slope)
  Altrimenti transizione, risolta dal contesto: dopo una Fase 4 è Fase 1
  (base), dopo una Fase 2 è Fase 3 (top). Come nel libro: è la storia
  recente a distinguere una base da un top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STAGE_NAMES = {1: "Base", 2: "Avanzata", 3: "Top", 4: "Declino"}


def classify_stages(enriched: pd.DataFrame, flat_slope: float) -> pd.Series:
    close = enriched["adj_close"].to_numpy()
    ma = enriched["ma"].to_numpy()
    slope = enriched["ma_slope"].to_numpy()
    n = len(enriched)
    stages = np.zeros(n, dtype=int)  # 0 = indeterminato (MA non matura)

    prev_trend = 0
    for t in range(n):
        if np.isnan(ma[t]) or np.isnan(slope[t]):
            continue
        above = close[t] > ma[t]
        rising = slope[t] > flat_slope
        falling = slope[t] < -flat_slope
        if above and rising:
            stages[t] = 2
            prev_trend = 2
        elif (not above) and falling:
            stages[t] = 4
            prev_trend = 4
        else:
            stages[t] = 1 if prev_trend in (0, 4) else 3
    return pd.Series(stages, index=enriched.index, name="stage")


def base_features(enriched: pd.DataFrame, stages: pd.Series,
                  gap_tolerance: int = 1, decline_lookback: int = 26,
                  vol_front_frac: float = 0.6) -> pd.DataFrame:
    """Per ogni settimana in Fase 1: durata, resistenza (max dei massimi),
    minimo e profondità della base in corso. Solo dati <= t (no look-ahead).

    gap_tolerance: la base di Weinstein è una STRUTTURA di prezzo, non una
    sequenza ininterrotta di etichette — una singola settimana rumorosa fuori
    Fase 1 non cancella una base di sei mesi. Tolleriamo fino a
    `gap_tolerance` settimane consecutive non-Fase-1 dentro una base."""
    # High/Low AGGIUSTATI: devono stare sulla stessa scala di adj_close, che
    # il segnale usa per il confronto `close > resistance`. Su un titolo con
    # split, gli high grezzi vivono su una scala diversa e il breakout non
    # scatta mai. Fallback ai grezzi se enrich_weekly non è passato.
    high = enriched["adj_high" if "adj_high" in enriched else "high"].to_numpy()
    low = enriched["adj_low" if "adj_low" in enriched else "low"].to_numpy()
    close = enriched["adj_close"].to_numpy().astype(float)
    # volume opzionale: se manca (fixture/enriched minimale), base_vol_ratio
    # resta NaN e il filtro di contrazione volume semplicemente non si applica.
    vol = (enriched["volume"].to_numpy().astype(float)
           if "volume" in enriched else np.full(len(enriched), np.nan))
    st = stages.to_numpy()
    n = len(st)
    base_len = np.zeros(n, dtype=int)
    resistance = np.full(n, np.nan)
    base_low = np.full(n, np.nan)
    after_decline = np.zeros(n, dtype=bool)
    # base_vol_ratio: volume medio nella PRIMA parte della base (i primi
    # `vol_front_frac` della sua durata = il dry-up iniziale/centrale) / volume
    # medio nelle `decline_lookback` settimane PRIMA della base. Weinstein: nella
    # base il volume si prosciuga (< 1), poi RIPRENDE avvicinandosi al breakout
    # (accumulazione). Misurare solo la prima parte isola il dry-up dalla ripresa
    # pre-breakout, che la media cumulativa includerebbe (caso MDSO). Solo dati <= t.
    base_vol_ratio = np.full(n, np.nan)
    # base_obv: accumulo/distribuzione NELLA base. OBV = somma del volume con
    # segno (+ nelle settimane di rialzo, − in quelle di ribasso), normalizzato
    # sul volume totale della base → [−1, +1]. > 0 = più volume sugli up = ACCUMULO
    # (Wyckoff/Weinstein). È il secondo livello: un titolo può non avere il dry-up
    # ma mostrare accumulo, e uno scanner vuole vederlo. Solo dati <= t.
    base_obv = np.full(n, np.nan)

    run = 0
    gap = 0
    res = np.nan
    lo = np.nan
    base_valid = False        # requisito A: la base corrente viene da un declino?
    ref_vol = np.nan          # volume medio pre-base (riferimento)
    base_vols: list[float] = []   # volumi delle settimane della base corrente
    obv_net = 0.0             # volume con segno accumulato nella base
    tot_vol = 0.0             # volume totale nella base (per normalizzare)

    def _add_vol(idx: int) -> None:
        nonlocal obv_net, tot_vol
        if not np.isfinite(vol[idx]):
            return
        base_vols.append(vol[idx])
        if idx >= 1 and np.isfinite(close[idx]) and np.isfinite(close[idx - 1]):
            sgn = 1.0 if close[idx] > close[idx - 1] else (-1.0 if close[idx] < close[idx - 1] else 0.0)
            obv_net += sgn * vol[idx]
            tot_vol += vol[idx]

    for t in range(n):
        if st[t] == 1:
            if run == 0:
                res, lo = high[t], low[t]
                # accumulazione DOPO declino: c'è stata una Fase 4 nelle
                # ultime `decline_lookback` settimane prima dell'inizio base?
                lo_start = max(0, t - decline_lookback)
                base_valid = bool((st[lo_start:t] == 4).any())
                # volume di riferimento: media nelle settimane prima della base
                pre = vol[lo_start:t]
                pre = pre[np.isfinite(pre)]
                ref_vol = float(pre.mean()) if len(pre) else np.nan
                base_vols = []
                obv_net, tot_vol = 0.0, 0.0
            else:
                res = max(res, high[t])
                lo = min(lo, low[t])
            run += 1
            gap = 0
            _add_vol(t)
        elif run > 0 and gap < gap_tolerance:
            gap += 1                      # settimana rumorosa dentro la base
            res = max(res, high[t])
            lo = min(lo, low[t])
            run += 1
            _add_vol(t)
        else:
            run, gap, res, lo = 0, 0, np.nan, np.nan
            base_valid = False
            ref_vol, base_vols = np.nan, []
            obv_net, tot_vol = 0.0, 0.0
        base_len[t] = run
        resistance[t] = res
        base_low[t] = lo
        after_decline[t] = base_valid
        # media sulla PRIMA parte della base (dry-up), esclusa la ripresa finale
        k_front = max(1, round(len(base_vols) * vol_front_frac))
        core = base_vols[:k_front] if base_vols else []
        if core and np.isfinite(ref_vol) and ref_vol > 0:
            base_vol_ratio[t] = (sum(core) / len(core)) / ref_vol
        if tot_vol > 0:
            base_obv[t] = obv_net / tot_vol

    out = pd.DataFrame({"base_len": base_len, "resistance": resistance,
                        "base_low": base_low, "after_decline": after_decline,
                        "base_vol_ratio": base_vol_ratio, "base_obv": base_obv},
                       index=enriched.index)
    out["base_depth"] = (out["resistance"] - out["base_low"]) / out["base_low"]
    return out


def top_features(enriched: pd.DataFrame, stages: pd.Series,
                 gap_tolerance: int = 1, advance_lookback: int = 26,
                 vol_front_frac: float = 0.6) -> pd.DataFrame:
    """Simmetrico di base_features per lo SHORT. Per ogni settimana in Fase 3
    (distribuzione/top): durata, supporto (min dei minimi), massimo e altezza
    del top. Il segnale short scatta quando il prezzo rompe SOTTO il supporto.

    advance_lookback: la Fase 3 di Weinstein è DISTRIBUZIONE DOPO UN AVANZAMENTO.
    Un top valido deve essere preceduto da una Fase 2 (rialzo) nelle ultime
    `advance_lookback` settimane — simmetrico di after_decline. Senza, si
    etichettano come 'top' anche le pause dentro un trend ribassista.

    Volumi speculari al long: `top_vol_ratio` (volume medio prima parte del top /
    volume pre-top) e `top_obv` (OBV nel top: <0 = DISTRIBUZIONE, più volume sui
    ribassi). Congelati come il supporto per essere usati al breakdown.
    """
    high = enriched["adj_high" if "adj_high" in enriched else "high"].to_numpy()
    low = enriched["adj_low" if "adj_low" in enriched else "low"].to_numpy()
    close = enriched["adj_close"].to_numpy().astype(float)
    vol = (enriched["volume"].to_numpy().astype(float)
           if "volume" in enriched else np.full(len(enriched), np.nan))
    st = stages.to_numpy()
    n = len(st)
    top_len = np.zeros(n, dtype=int)
    support = np.full(n, np.nan)       # min dei minimi del top (da rompere sotto)
    top_high = np.full(n, np.nan)      # max dei massimi del top
    after_advance = np.zeros(n, dtype=bool)
    top_vol_ratio = np.full(n, np.nan)
    top_obv = np.full(n, np.nan)

    run = 0
    gap = 0
    sup = np.nan
    hi = np.nan
    top_valid = False
    ref_vol = np.nan
    top_vols: list[float] = []
    obv_net = 0.0
    tot_vol = 0.0
    vr_frozen = np.nan
    obv_frozen = np.nan

    def _add_vol(idx: int) -> None:
        nonlocal obv_net, tot_vol
        if not np.isfinite(vol[idx]):
            return
        top_vols.append(vol[idx])
        if idx >= 1 and np.isfinite(close[idx]) and np.isfinite(close[idx - 1]):
            sgn = 1.0 if close[idx] > close[idx - 1] else (-1.0 if close[idx] < close[idx - 1] else 0.0)
            obv_net += sgn * vol[idx]
            tot_vol += vol[idx]
    # support_est = supporto STABILITO del top: il minimo dei minimi delle
    # settimane PRECEDENTI dentro il top, escludendo la corrente. È il livello
    # che il prezzo deve rompere. Se includessimo la settimana corrente, il
    # prezzo non potrebbe mai stare "sotto il proprio minimo" e il breakdown
    # non scatterebbe mai. Persiste `support_persist` settimane dopo la fine
    # del top, perché il breakdown avviene proprio in transizione a Fase 4.
    support_est = np.full(n, np.nan)
    top_height_est = np.full(n, np.nan)
    after_advance_est = np.zeros(n, dtype=bool)
    support_persist = 6
    persist_left = 0
    sup_frozen = np.nan
    hi_frozen = np.nan
    valid_frozen = False
    for t in range(n):
        # PRIMA di aggiornare col dato di t: registro il supporto stabilito
        # finora (solo settimane < t nel top), che è ciò che il prezzo di t
        # può rompere senza look-ahead
        if run >= 8:
            sup_frozen, hi_frozen, valid_frozen = sup, hi, top_valid
            persist_left = support_persist
            # congela anche i volumi del top: dry-up (prima parte) e OBV
            k_front = max(1, round(len(top_vols) * vol_front_frac))
            core = top_vols[:k_front] if top_vols else []
            vr_frozen = ((sum(core) / len(core)) / ref_vol
                         if core and np.isfinite(ref_vol) and ref_vol > 0 else np.nan)
            obv_frozen = (obv_net / tot_vol) if tot_vol > 0 else np.nan

        if st[t] == 3:
            if run == 0:
                sup, hi = low[t], high[t]
                lo_start = max(0, t - advance_lookback)
                top_valid = bool((st[lo_start:t] == 2).any())
                pre = vol[lo_start:t]
                pre = pre[np.isfinite(pre)]
                ref_vol = float(pre.mean()) if len(pre) else np.nan
                top_vols = []
                obv_net, tot_vol = 0.0, 0.0
            else:
                sup = min(sup, low[t])
                hi = max(hi, high[t])
            run += 1
            gap = 0
            _add_vol(t)
        elif run > 0 and gap < gap_tolerance:
            gap += 1
            sup = min(sup, low[t])
            hi = max(hi, high[t])
            run += 1
            _add_vol(t)
        else:
            run, gap, sup, hi = 0, 0, np.nan, np.nan
            top_valid = False
            ref_vol, top_vols = np.nan, []
            obv_net, tot_vol = 0.0, 0.0
        top_len[t] = run
        support[t] = sup
        top_high[t] = hi
        after_advance[t] = top_valid

        # il supporto stabilito (congelato) vale per il breakdown, e persiste
        if persist_left > 0 and np.isfinite(sup_frozen):
            support_est[t] = sup_frozen
            top_height_est[t] = (hi_frozen - sup_frozen) / sup_frozen
            after_advance_est[t] = valid_frozen
            top_vol_ratio[t] = vr_frozen
            top_obv[t] = obv_frozen
            if st[t] != 3:            # fuori dal top: consuma la persistenza
                persist_left -= 1

    out = pd.DataFrame({"top_len": top_len, "support": support,
                        "top_high": top_high, "after_advance": after_advance,
                        "support_est": support_est,
                        "top_height_est": top_height_est,
                        "after_advance_est": after_advance_est,
                        "top_vol_ratio": top_vol_ratio, "top_obv": top_obv},
                       index=enriched.index)
    out["top_height"] = (out["top_high"] - out["support"]) / out["support"]
    return out
