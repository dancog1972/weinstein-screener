#!/usr/bin/env python3
"""Decomposizione del metodo Weinstein: dove aggiunge o distrugge valore.

Misura 1 — QUALITÀ PURA DEGLI INGRESSI.
Ogni segnale porta con sé i forward returns: quanto avresti guadagnato COMPRANDO
al segnale e tenendo N settimane, SENZA stop e SENZA filtri di portafoglio. È la
qualità dell'ingresso isolata da tutto il resto.

La domanda: questi rendimenti battono il mercato (exc > 0)?
- Se sì: gli ingressi di Weinstein sono buoni, il valore si perde a valle
  (stop che tagliano i vincitori, market filter che tiene fuori nei momenti
  giusti). Guarderemo lì.
- Se no: è la selezione stessa a non funzionare. Nessuno stop può salvare
  ingressi che in media perdono contro il mercato.

  python analyze.py --config config_us_liquidity.yaml --measure entries
"""
from __future__ import annotations

import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.config import load_config
from src.data_store import DataStore
from src.providers import make_provider


def measure_entries(cfg: dict, store: DataStore) -> None:
    """Misura 1: qualità pura degli ingressi via forward returns."""
    print("Calcolo dei segnali (fase pesante, alcuni minuti)...\n")
    res = run_backtest(cfg, store, progress=True)
    ss = res.signal_stats
    if ss.empty:
        print("Nessun segnale.")
        return

    horizons = cfg["backtest"]["horizons_weeks"]
    n = len(ss)
    print("\n" + "=" * 66)
    print(f"MISURA 1 — QUALITÀ PURA DEGLI INGRESSI ({n} segnali)")
    print("Rendimento tenendo N settimane, senza stop né filtri di portafoglio")
    print("=" * 66)
    print(f"\n{'orizz.':>7} {'rend medio':>11} {'vs mercato':>11} "
          f"{'% positivi':>11} {'% batte mkt':>12} {'mediana':>9}")
    print("-" * 66)

    verdicts = []
    for h in horizons:
        fwd = ss[f"fwd_{h}w"].dropna()
        exc = ss[f"exc_{h}w"].dropna()
        if len(fwd) == 0:
            continue
        # Sanificazione: un forward return oltre +900% o sotto -100% è
        # corruzione dati residua (split-adjustment sballato che esplode a
        # orizzonti lunghi), non un vero rendimento. Lo scarto dalle medie —
        # altrimenti un solo segnale corrotto le rende assurde (+7800%).
        good = ss[f"fwd_{h}w"].notna() & ss[f"fwd_{h}w"].between(-1.0, 9.0)
        n_drop = int((ss[f"fwd_{h}w"].notna() & ~good).sum())
        fwd_clean = ss.loc[good, f"fwd_{h}w"]
        exc_clean = ss.loc[good, f"exc_{h}w"].dropna()
        if len(fwd_clean) == 0:
            continue
        pct_pos = (fwd_clean > 0).mean() * 100
        pct_beat = (exc_clean > 0).mean() * 100
        flag = f"  (scartati {n_drop} outlier)" if n_drop else ""
        print(f"{h:>5}w {fwd_clean.mean()*100:>+10.1f}% {exc_clean.mean()*100:>+10.1f}% "
              f"{pct_pos:>10.0f}% {pct_beat:>11.0f}% {fwd_clean.median()*100:>+8.1f}%{flag}")
        verdicts.append((h, exc_clean.mean(), pct_beat, fwd_clean.median()))

    print("=" * 66)

    # verdetto sull'orizzonte più lungo, basato su MEDIANA e % che batte il
    # mercato (robuste agli outlier), non sulla media (che pochi grandi
    # vincitori possono gonfiare)
    if verdicts:
        h, exc_mean, pct_beat, med = verdicts[-1]
        print(f"\nA {h} settimane (mediana {med*100:+.1f}%, "
              f"{pct_beat:.0f}% batte il mercato):")
        if pct_beat > 55 and med > 0.02:
            print("  ✓ Gli ingressi BATTONO il mercato più spesso del caso.")
            print("    La selezione ha valore; il problema è a VALLE (stop, filtri).")
            print("    Prossimo: misurare i trade REALI vs questi forward returns.")
        elif pct_beat < 48:
            print("  ✗ Meno della metà dei segnali batte il mercato: come una")
            print("    monetina. Gli ingressi NON aggiungono selettività — il")
            print("    problema è a MONTE, negli ingressi stessi. Nessuno stop")
            print("    può creare valore che nella selezione non c'è.")
        else:
            print("  ~ Gli ingressi sono a malapena distinguibili dal caso")
            print(f"    ({pct_beat:.0f}% batte il mercato, mediana {med*100:+.1f}%).")
            print("    La selezione aggiunge poco o nulla.")

    # confronto con un ingresso casuale sarà la misura successiva
    print("\nNota: questo misura gli ingressi in ISOLAMENTO. Il confronto col")
    print("caso (gemello) resta il metro del picking complessivo.")


def measure_characteristics(cfg: dict, store: DataStore) -> None:
    """Misura 2: quale CARATTERISTICA del segnale predice i vincitori?
    Divide i segnali in fasce per Mansfield, lunghezza base, profondità e volume
    ratio, e per ogni fascia misura quanti battono il mercato. Se una fascia
    supera nettamente il 50%, quella caratteristica è un selettore utile: il
    filtro che a Weinstein-base manca."""
    print("Calcolo dei segnali (fase pesante, alcuni minuti)...\n")
    res = run_backtest(cfg, store, progress=True)
    ss = res.signal_stats
    if ss.empty or len(ss) < 20:
        print(f"Troppi pochi segnali ({len(ss)}) per un'analisi per fasce.")
        return

    horizons = cfg["backtest"]["horizons_weeks"]
    h = horizons[-1]      # orizzonte più lungo (52w): dove il segnale conta
    # sanifico gli outlier di forward/excess come nella misura 1
    good = ss[f"fwd_{h}w"].notna() & ss[f"fwd_{h}w"].between(-1.0, 9.0)
    d = ss[good].copy()
    d["beat"] = d[f"exc_{h}w"] > 0      # batte il mercato a h settimane?

    n = len(d)
    base_beat = d["beat"].mean() * 100
    print("\n" + "=" * 70)
    print(f"MISURA 2 — QUALE CARATTERISTICA PREDICE I VINCITORI ({n} segnali, {h}w)")
    print("=" * 70)
    print(f"Baseline: {base_beat:.0f}% dei segnali batte il mercato a {h} settimane.")
    print("Cerchiamo fasce che superano NETTAMENTE questa baseline.\n")

    characteristics = [
        ("mansfield", "Forza relativa (Mansfield)"),
        ("base_len", "Lunghezza base (settimane)"),
        ("base_depth", "Profondità base"),
        ("vol_ratio", "Volume ratio al breakout"),
    ]

    findings = []
    for col, label in characteristics:
        if col not in d.columns:
            continue
        vals = d[col].dropna()
        if len(vals) < 20:
            continue
        # divido in quartili: cerco se l'ultimo quartile (valori alti) batte
        # il mercato più del primo (valori bassi)
        try:
            d["_q"] = pd.qcut(d[col].rank(method="first"), 4,
                              labels=["Q1 basso", "Q2", "Q3", "Q4 alto"])
        except ValueError:
            continue
        print(f"── {label} ──")
        print(f"   {'fascia':<10} {'range':>20} {'% batte mkt':>12} {'mediana ret':>12}")
        q_beats = {}
        for q in ["Q1 basso", "Q2", "Q3", "Q4 alto"]:
            sub = d[d["_q"] == q]
            if len(sub) == 0:
                continue
            beat_pct = sub["beat"].mean() * 100
            med_ret = sub[f"fwd_{h}w"].median() * 100
            lo, hi = sub[col].min(), sub[col].max()
            q_beats[q] = beat_pct
            print(f"   {q:<10} {f'{lo:.1f}–{hi:.1f}':>20} {beat_pct:>11.0f}% {med_ret:>+11.1f}%")
        # la caratteristica è utile se Q4 batte Q1 di almeno 10 punti
        if "Q4 alto" in q_beats and "Q1 basso" in q_beats:
            spread = q_beats["Q4 alto"] - q_beats["Q1 basso"]
            if spread >= 10:
                findings.append((label, spread, q_beats["Q4 alto"]))
                print(f"   → i valori ALTI battono i bassi di {spread:.0f} punti "
                      f"({q_beats['Q4 alto']:.0f}% vs {q_beats['Q1 basso']:.0f}%)")
            elif spread <= -10:
                findings.append((label, spread, q_beats["Q1 basso"]))
                print(f"   → i valori BASSI battono gli alti di {-spread:.0f} punti "
                      f"({q_beats['Q1 basso']:.0f}% vs {q_beats['Q4 alto']:.0f}%)")
            else:
                print(f"   → nessuna separazione netta (spread {spread:+.0f} punti)")
        print()

    print("=" * 70)
    if findings:
        findings.sort(key=lambda x: abs(x[1]), reverse=True)
        print("CARATTERISTICHE PROMETTENTI (separano vincitori da perdenti):")
        for label, spread, best in findings:
            print(f"  • {label}: fascia migliore batte il mercato nel {best:.0f}% dei casi")
        print("\nProssimo passo: usare la caratteristica migliore come FILTRO")
        print("aggiuntivo e rimisurare. Se il filtro alza la % oltre il 55-60%,")
        print("hai trovato il selettore che a Weinstein-base manca.")
    else:
        print("NESSUNA caratteristica separa i vincitori dai perdenti.")
        print("Le fasce alte e basse battono il mercato più o meno uguale.")
        print("Implica: dentro i segnali di Weinstein non c'è un sottoinsieme")
        print("identificabile in anticipo che batte il mercato. La selezione")
        print("long su questo universo è un vicolo cieco — resta lo short (Fase 4).")


def measure_entries_short(cfg: dict, store: DataStore) -> None:
    """Misura 1 per lo SHORT. I breakdown di Weinstein (Fase 3→4), tenuti short,
    guadagnano quando il titolo scende PIÙ del mercato?

    Attenzione ai segni: per lo short un rendimento negativo del titolo è un
    GUADAGNO. Il rendimento short a N settimane è -(variazione del titolo).
    'Batte il mercato' significa: lo short ha reso più che shortare l'indice,
    cioè il titolo è sceso più del mercato (o salito meno)."""
    import numpy as np
    import pandas as pd
    from src.indicators import enrich_weekly, to_weekly
    from src.signals import prepare_ticker_short, detect_signals_short
    from src.stages import classify_stages
    from src.backtest import load_universe

    bt, st_cfg = cfg["backtest"], cfg["stages"]
    start, end = bt["start"], bt["end"]
    horizons = bt["horizons_weeks"]
    universe = load_universe(cfg)

    print("Calcolo dei segnali SHORT (fase pesante, alcuni minuti)...\n")
    # benchmark per i forward returns e per il market filter (Fase 4)
    rows = []
    n_sig = 0
    for market, spec in universe.items():
        b_daily = store.get_with_warmup(spec["benchmark"], start, end)
        b_wk = enrich_weekly(to_weekly(b_daily), None, st_cfg["ma_weeks"],
                             st_cfg["slope_lookback"])
        mkt_stage = classify_stages(b_wk, st_cfg["flat_slope"])
        tickers = spec["tickers"]
        for i, tk in enumerate(tickers):
            try:
                d = store.get_with_warmup(tk, start, end)
                wk = enrich_weekly(to_weekly(d), b_wk, st_cfg["ma_weeks"],
                                   st_cfg["slope_lookback"])
            except Exception:  # noqa: BLE001
                continue
            if wk.empty or wk["ma"].notna().sum() < st_cfg["ma_weeks"]:
                continue
            df = prepare_ticker_short(wk, st_cfg["flat_slope"])
            sigs = detect_signals_short(tk, df, mkt_stage, cfg["signal"],
                                        cfg["exits"], st_cfg["flat_slope"])
            sigs = [s for s in sigs if pd.Timestamp(start) <= s.date <= pd.Timestamp(end)]
            for s in sigs:
                idx = df.index.get_loc(s.date)
                b_idx = b_wk.index.get_indexer([s.date], method="ffill")[0]
                row = {"ticker": tk, "date": s.date, "mansfield": s.mansfield,
                       "base_len": s.base_len, "vol_ratio": s.vol_ratio}
                for h in horizons:
                    if idx + h < len(df) and 0 <= b_idx and b_idx + h < len(b_wk):
                        # rendimento del TITOLO tenendo h settimane
                        r_titolo = df["adj_close"].iloc[idx + h] / s.entry - 1.0
                        r_bench = b_wk["adj_close"].iloc[b_idx + h] / b_wk["adj_close"].iloc[b_idx] - 1.0
                        # SHORT: guadagno = -(rendimento). Batte il mercato se
                        # lo short del titolo rende più dello short dell'indice,
                        # cioè se il titolo scende più del mercato: -r_tit > -r_ben
                        row[f"short_{h}w"] = -r_titolo
                        row[f"short_exc_{h}w"] = -r_titolo - (-r_bench)
                    else:
                        row[f"short_{h}w"] = np.nan
                        row[f"short_exc_{h}w"] = np.nan
                rows.append(row)
            n_sig += len(sigs)
            if i and i % 2000 == 0:
                print(f"  ...{i}/{len(tickers)} titoli · {n_sig} segnali short", flush=True)

    ss = pd.DataFrame(rows)
    if ss.empty:
        print("\nNessun segnale short rilevato.")
        return

    n = len(ss)
    print("\n" + "=" * 66)
    print(f"MISURA 1-SHORT — QUALITÀ DEGLI INGRESSI SHORT ({n} segnali)")
    print("Rendimento SHORT tenendo N settimane (titolo che scende = guadagno)")
    print("=" * 66)
    print(f"\n{'orizz.':>7} {'rend short':>11} {'vs mercato':>11} "
          f"{'% positivi':>11} {'% batte mkt':>12} {'mediana':>9}")
    print("-" * 66)
    verdicts = []
    for h in horizons:
        good = ss[f"short_{h}w"].notna() & ss[f"short_{h}w"].between(-1.0, 9.0)
        n_drop = int((ss[f"short_{h}w"].notna() & ~good).sum())
        sh = ss.loc[good, f"short_{h}w"]
        exc = ss.loc[good, f"short_exc_{h}w"].dropna()
        if len(sh) == 0:
            continue
        pct_pos = (sh > 0).mean() * 100
        pct_beat = (exc > 0).mean() * 100
        flag = f"  (scartati {n_drop})" if n_drop else ""
        print(f"{h:>5}w {sh.mean()*100:>+10.1f}% {exc.mean()*100:>+10.1f}% "
              f"{pct_pos:>10.0f}% {pct_beat:>11.0f}% {sh.median()*100:>+8.1f}%{flag}")
        verdicts.append((h, exc.mean(), pct_beat, sh.median()))

    print("=" * 66)
    if verdicts:
        h, exc_mean, pct_beat, med = verdicts[-1]
        print(f"\nA {h} settimane (mediana short {med*100:+.1f}%, "
              f"{pct_beat:.0f}% batte il mercato):")
        if pct_beat > 55 and med > 0.02:
            print("  ✓ Gli SHORT battono il mercato più spesso del caso.")
            print("    La Fase 4 seleziona titoli che scendono più del mercato:")
            print("    lo short di Weinstein ha valore. Vale la pena costruire")
            print("    il portafoglio short completo (costi di prestito, target).")
        elif pct_beat < 48:
            print("  ✗ Meno della metà degli short batte il mercato: come una")
            print("    monetina. Anche i breakdown di Fase 4 non selezionano.")
            print("    Come il long, lo short non aggiunge valore su questo universo.")
        else:
            print("  ~ Gli short sono a malapena distinguibili dal caso.")
            print("    La Fase 4 aggiunge poca selettività.")
    print("\nNota: misura gli ingressi short in ISOLAMENTO, senza stop, costi")
    print("di prestito o profit target. È il test preliminare che decide se")
    print("vale la pena costruire il portafoglio short completo.")


def measure_mansfield(cfg: dict, store: DataStore) -> None:
    """Test sul filtro Mansfield. Verifica l'intuizione: un titolo che esce da
    una Fase 1 (declino + base) ha Mansfield ancora negativo perché la media a
    52 settimane porta il peso del declino. Il filtro 'mansfield >= 0' potrebbe
    escludere proprio i migliori candidati — quelli che stanno INVERTENDO.

    Test A: il filtro Mansfield, acceso vs spento, cambia la qualità?
    Test B: conta più il LIVELLO (positivo/negativo) o la DIREZIONE (sale/scende)?
    Separa i segnali in 4 quadranti e misura quale batte il mercato."""
    import numpy as np
    import pandas as pd
    from src.indicators import enrich_weekly, to_weekly
    from src.signals import prepare_ticker, detect_signals
    from src.stages import classify_stages
    from src.backtest import load_universe

    bt, st_cfg = cfg["backtest"], cfg["stages"]
    start, end = bt["start"], bt["end"]
    horizons = bt["horizons_weeks"]
    h = horizons[-1]
    universe = load_universe(cfg)

    print("Calcolo segnali con e senza filtro Mansfield (fase pesante)...\n")
    # genero i segnali DISATTIVANDO il filtro Mansfield, e per ognuno registro
    # sia il livello sia la direzione, più il forward return
    cfg_no_mans = {**cfg, "signal": {**cfg["signal"], "mansfield_min": -9999.0,
                                     "require_mansfield_rising": False}}

    rows = []
    for market, spec in universe.items():
        b_daily = store.get_with_warmup(spec["benchmark"], start, end)
        b_wk = enrich_weekly(to_weekly(b_daily), None, st_cfg["ma_weeks"],
                             st_cfg["slope_lookback"])
        mkt_stage = classify_stages(b_wk, st_cfg["flat_slope"])
        tickers = spec["tickers"]
        for i, tk in enumerate(tickers):
            try:
                d = store.get_with_warmup(tk, start, end)
                wk = enrich_weekly(to_weekly(d), b_wk, st_cfg["ma_weeks"],
                                   st_cfg["slope_lookback"])
            except Exception:  # noqa: BLE001
                continue
            if wk.empty or wk["ma"].notna().sum() < st_cfg["ma_weeks"]:
                continue
            df = prepare_ticker(wk, st_cfg["flat_slope"])
            # segnali SENZA filtro mansfield (ma con mansfield ancora richiesto
            # crescente? no: disattivo anche quello per vedere tutto)
            cfg_sig = {**cfg_no_mans["signal"]}
            sigs = detect_signals(tk, df, mkt_stage, cfg_sig, cfg["exits"],
                                  st_cfg["flat_slope"])
            sigs = [s for s in sigs if pd.Timestamp(start) <= s.date <= pd.Timestamp(end)]
            for s in sigs:
                idx = df.index.get_loc(s.date)
                b_idx = b_wk.index.get_indexer([s.date], method="ffill")[0]
                if not (idx + h < len(df) and 0 <= b_idx and b_idx + h < len(b_wk)):
                    continue
                r_tit = df["adj_close"].iloc[idx + h] / s.entry - 1.0
                r_ben = b_wk["adj_close"].iloc[b_idx + h] / b_wk["adj_close"].iloc[b_idx] - 1.0
                # direzione del mansfield: confronto con 4 settimane prima
                mans_now = s.mansfield
                mans_prev = df["mansfield"].iloc[idx - 4] if idx >= 4 else np.nan
                mans_dir = mans_now - mans_prev if pd.notna(mans_prev) else np.nan
                rows.append({"mansfield": mans_now, "mans_dir": mans_dir,
                             "fwd": r_tit, "exc": r_tit - r_ben})
            if i and i % 3000 == 0:
                print(f"  ...{i}/{len(tickers)} titoli", flush=True)

    d = pd.DataFrame(rows)
    d = d[d["fwd"].between(-1.0, 9.0)].dropna(subset=["mansfield", "mans_dir"])
    if len(d) < 40:
        print(f"Troppi pochi segnali ({len(d)}).")
        return
    d["beat"] = d["exc"] > 0

    # ---- TEST A: filtro acceso vs spento -----------------------------------
    print("\n" + "=" * 68)
    print(f"TEST A — IL FILTRO MANSFIELD AIUTA? (orizzonte {h}w)")
    print("=" * 68)
    tutti = d["beat"].mean() * 100
    con_filtro = d[d["mansfield"] >= 0]["beat"].mean() * 100
    senza = d[d["mansfield"] < 0]["beat"].mean() * 100
    n_pos = (d["mansfield"] >= 0).sum()
    n_neg = (d["mansfield"] < 0).sum()
    print(f"  Segnali con Mansfield POSITIVO (filtro li TIENE): "
          f"{n_pos} segnali, {con_filtro:.0f}% batte il mercato")
    print(f"  Segnali con Mansfield NEGATIVO (filtro li SCARTA): "
          f"{n_neg} segnali, {senza:.0f}% batte il mercato")
    print(f"  Tutti insieme: {len(d)} segnali, {tutti:.0f}% batte il mercato")
    if senza > con_filtro + 3:
        print("\n  → I segnali SCARTATI dal filtro battono il mercato PIÙ di quelli")
        print("    tenuti. Il filtro Mansfield DANNEGGIA: esclude vincitori.")
        print("    L'intuizione è confermata.")
    elif con_filtro > senza + 3:
        print("\n  → I segnali tenuti battono il mercato più di quelli scartati.")
        print("    Il filtro Mansfield AIUTA: la forza relativa seleziona.")
    else:
        print("\n  → Nessuna differenza netta: il filtro Mansfield è ininfluente.")

    # ---- TEST B: livello × direzione (4 quadranti) -------------------------
    print("\n" + "=" * 68)
    print(f"TEST B — LIVELLO o DIREZIONE? Quadranti Mansfield ({h}w)")
    print("=" * 68)
    print("  L'intuizione: un titolo debole ma che sta GIRANDO (Mansfield")
    print("  negativo ma in salita) potrebbe battere uno forte ma in calo.\n")
    quads = [
        ("Positivo & in salita", (d["mansfield"] >= 0) & (d["mans_dir"] > 0)),
        ("Positivo & in calo   ", (d["mansfield"] >= 0) & (d["mans_dir"] <= 0)),
        ("Negativo & in salita ", (d["mansfield"] < 0) & (d["mans_dir"] > 0)),
        ("Negativo & in calo   ", (d["mansfield"] < 0) & (d["mans_dir"] <= 0)),
    ]
    print(f"  {'quadrante':<24} {'n':>5} {'% batte mkt':>12} {'mediana ret':>12}")
    print("  " + "-" * 56)
    results = []
    for label, mask in quads:
        sub = d[mask]
        if len(sub) < 5:
            print(f"  {label:<24} {len(sub):>5}   (troppo pochi)")
            continue
        beat = sub["beat"].mean() * 100
        med = sub["fwd"].median() * 100
        results.append((label, len(sub), beat, med))
        print(f"  {label:<24} {len(sub):>5} {beat:>11.0f}% {med:>+11.1f}%")
    print("  " + "-" * 56)
    if results:
        best = max(results, key=lambda x: x[2])
        print(f"\n  Quadrante migliore: {best[0].strip()} ({best[2]:.0f}% batte il mercato)")
        # l'intuizione è confermata se "negativo & in salita" batte "positivo & in calo"
        neg_up = next((r for r in results if "Negativo & in salita" in r[0]), None)
        pos_dn = next((r for r in results if "Positivo & in calo" in r[0]), None)
        if neg_up and pos_dn:
            if neg_up[2] > pos_dn[2] + 5:
                print(f"  → I titoli DEBOLI ma in ripresa ({neg_up[2]:.0f}%) battono i FORTI")
                print(f"    ma in calo ({pos_dn[2]:.0f}%). La DIREZIONE conta più del livello:")
                print("    l'intuizione è confermata. Weinstein dovrebbe pesare la")
                print("    direzione del Mansfield, non solo il segno.")
            elif pos_dn[2] > neg_up[2] + 5:
                print(f"  → I titoli FORTI battono i deboli anche se questi salgono.")
                print("    Il livello conta più della direzione: il filtro classico regge.")
            else:
                print("  → Livello e direzione si equivalgono: nessuno domina.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--measure", default="entries",
                    choices=["entries", "characteristics", "entries-short", "mansfield"],
                    help="quale misura di decomposizione")
    args = ap.parse_args()
    cfg = load_config(args.config)
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    if args.measure == "entries":
        measure_entries(cfg, store)
    elif args.measure == "characteristics":
        measure_characteristics(cfg, store)
    elif args.measure == "entries-short":
        measure_entries_short(cfg, store)
    elif args.measure == "mansfield":
        measure_mansfield(cfg, store)


if __name__ == "__main__":
    main()
