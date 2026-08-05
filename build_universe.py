#!/usr/bin/env python3
"""Costruisce l'universo storico e scarica i prezzi in cache locale.

Uso:
  export EODHD_API_KEY=la_tua_chiave

  # 1) scarica la composizione storica dell'indice (1 chiamata)
  python build_universe.py constituents --index GSPC.INDX

  # 2) scarica i prezzi di TUTTI i ticker mai apparsi (~800 chiamate)
  python build_universe.py prices --index GSPC.INDX --start 2000-01-01

  # riprendi dopo un'interruzione: salta ciò che è già in cache
  python build_universe.py prices --index GSPC.INDX --start 2000-01-01

Lo scaricamento è idempotente e resiliente: ogni ticker va in un file Parquet
separato, i falliti finiscono in un log e si ritentano al giro successivo.
Interrompere con Ctrl+C non perde il lavoro fatto.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys

# UTF-8 su Windows: la console usa cp1252 e crasha sui simboli (→ × ✓ ⚠).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.data_store import DataStore
from src.providers import make_provider
from src.sectors import SECTOR_ETFS
from src.universe import PointInTimeUniverse

FAILED_LOG = "failed_tickers.json"


def _scrub(msg: str, limit: int = 140) -> str:
    """Rimuove il token API dai messaggi d'errore.

    requests include l'URL completo nelle sue eccezioni, e il nostro URL
    contiene `api_token=...`. Senza questa pulizia la chiave finisce nel file
    dei falliti, negli screenshot, nei log condivisi. È già successo.
    """
    msg = re.sub(r"(api_token=)[^&\s]+", r"\1***", msg)
    return msg[:limit]
SECTOR_ETF_LIST = sorted(SECTOR_ETFS.values())


def cmd_constituents(args) -> None:
    provider = make_provider("eodhd")
    print(f"Scarico la composizione storica di {args.index}...")
    uni = PointInTimeUniverse.from_provider(provider, args.index)
    path = uni.save(args.data_dir)
    s = uni.stats()
    print(f"\n✓ salvato in {path}")
    print(f"  ticker totali mai apparsi: {s['tickers_totali']}")
    print(f"  di cui delistati: {s['delistati']}  ← questi eliminano il survivorship bias")
    print(f"  ancora nell'indice oggi: {s['ancora_attivi']}")
    print(f"  copertura: {s['primo_ingresso']} → oggi")
    print(f"\nProssimo passo:\n  python build_universe.py prices --index {args.index} "
          f"--start {args.start}")


def cmd_prices(args) -> None:
    # Il file dell'universo si controlla PRIMA di costruire il provider:
    # se mancano sia il file sia la chiave API, l'errore utile è questo.
    up = Path(args.from_universe) if args.from_universe else None
    if up is not None and not up.exists():
        print(f"✗ Universo non trovato: {up}\n")
        print("  Va costruito prima, con un passo separato:")
        print("    python build_universe.py screen --exchange US --limit 800\n")
        print("  (`screen` scarica la lista dei ticker e la salva in quel file;")
        print("   `prices` poi ne scarica gli storici.)")
        raise SystemExit(1)

    provider = make_provider("eodhd")
    store = DataStore(args.data_dir, provider)

    if args.from_universe:
        # universo per liquidità (o qualunque lista salvata da `screen`)
        df = pd.read_parquet(up)
        tickers = sorted(df["ticker"].tolist())
        bench = args.benchmark or "SPY.US"
        extra = SECTOR_ETF_LIST if args.with_sector_etfs else []
        todo = [bench] + extra + [t for t in tickers if t != bench]
        print(f"Universo da {args.from_universe}: {len(tickers)} ticker")
    else:
        uni = PointInTimeUniverse.load(args.data_dir, args.index)
        tickers = uni.all_tickers()
        bench = args.benchmark or args.index
        todo = [bench] + [t for t in tickers if t != bench]

    root = Path(args.data_dir) / provider.name
    already = {p.stem.replace("_", ".") for p in root.glob("*.parquet")} if root.exists() else set()
    remaining = [t for t in todo if t.replace("/", "_") not in already]

    print(f"  già in cache: {len(todo) - len(remaining)} · da scaricare: {len(remaining)}")
    if not remaining:
        print("\n✓ cache completa, nulla da fare.")
        return
    est_min = len(remaining) * (provider.throttle_s + 0.35) / 60
    print(f"  tempo stimato: ~{est_min:.0f} min · chiamate: {len(remaining)}\n")
    if not args.yes:
        if input("procedo? [s/N] ").strip().lower() not in ("s", "y"):
            print("annullato."); return

    failed: list[dict] = []
    t0 = time.time()
    for i, tk in enumerate(remaining, 1):
        try:
            df = store.get_with_warmup(tk, args.start, args.end)
            status = f"{len(df):5d} barre"
        except Exception as e:  # noqa: BLE001
            # un titolo che fallisce non deve fermare 800 download
            failed.append({"ticker": tk, "error": _scrub(str(e))})
            status = f"FALLITO ({type(e).__name__})"
        el = time.time() - t0
        eta = el / i * (len(remaining) - i)
        print(f"  [{i:4d}/{len(remaining)}] {tk:14s} {status:24s} ETA {eta/60:4.1f} min", flush=True)

    if failed:
        p = Path(args.data_dir) / FAILED_LOG
        p.write_text(json.dumps(failed, indent=2), encoding="utf-8")
        print(f"\n⚠ {len(failed)} ticker falliti, elencati in {p}")
        print("  Rilancia lo stesso comando per ritentarli (i riusciti sono in cache).")
        print("  Cause tipiche: titolo delistato senza storico, ticker rinominato,")
        print("  quota API esaurita per oggi.")
    print(f"\n✓ completato in {(time.time()-t0)/60:.1f} min")


def cmd_screen(args) -> None:
    """Costruisce l'universo per liquidità dalla borsa indicata.

    Due stadi, perché la borsa `US` ha decine di migliaia di ticker e non ha
    senso scaricarne i prezzi tutti:
      1) filtro GROSSOLANO sui metadati (2 chiamate): solo Common Stock,
         niente ETF/fondi/preferred, niente OTC.
      2) il filtro di liquidità VERO si applica poi settimana per settimana
         dentro il backtest, sui prezzi effettivamente scaricati.

    I delistati sono inclusi: sono loro a eliminare il survivorship bias.
    """
    provider = make_provider("eodhd")
    print(f"Scarico la lista ticker di {args.exchange} (attivi + delistati)...")
    live = provider.exchange_symbols(args.exchange, delisted=False)
    dead = provider.exchange_symbols(args.exchange, delisted=True)
    print(f"  attivi: {len(live)} · delistati: {len(dead)}")

    df = pd.concat([live, dead], ignore_index=True)
    before = len(df)

    # stadio 1: solo azioni ordinarie, niente ETF/fondi/preferred/OTC
    if "type" in df.columns:
        df = df[df["type"].str.contains("Common Stock", case=False, na=False)]
    # `exchange_real` è la borsa vera (NYSE/NASDAQ/OTC...), mentre `ticker`
    # usa il codice di richiesta `.US`. Il filtro OTC ragiona sulla prima.
    otc_col = "exchange_real" if "exchange_real" in df.columns else "exchange"
    if otc_col in df.columns and args.exclude_otc:
        # EODHD marca il mercato OTC con PIÙ codici: OTC, OTCMKTS, OTCQB, OTCQX,
        # OTCGREY, e anche OTCCE e OTCBB. Una lista esplicita ne dimenticava due
        # (OTCCE/OTCBB) e 1.858 titoli OTC — pink-sheet, sub-penny, shell —
        # sfuggivano al filtro e inquinavano i segnali. Un prefisso "OTC" li
        # prende tutti; PINK/GREY sono gli altri nomi dei mercati non regolamentati.
        ex = df[otc_col].astype(str).str.upper()
        df = df[~(ex.str.startswith("OTC") | ex.isin(["PINK", "GREY"]))]
    if "currency" in df.columns and args.currency:
        df = df[df["currency"].astype(str).str.upper() == args.currency.upper()]
    df = df.drop_duplicates(subset=["ticker"])

    # Ticker "spazzatura" che il campo `type` non intercetta: warrant, unit,
    # right, preferred. Su NASDAQ hanno suffissi convenzionali (AACBR = right
    # di AACB). Scaricarne lo storico è spreco: non sono azioni ordinarie.
    junk = df["code"].astype(str).str.upper().str.match(r".*(W|WS|R|U|P|PR[A-Z]?)$")
    plain = df["code"].astype(str).str.len() <= 4
    df = df[~(junk & ~plain)]
    # `_old`: ticker riassegnato a un'altra azienda dopo un delisting. Sono
    # società DIVERSE, e il loro storico è spesso frammentario.
    df = df[~df["code"].astype(str).str.contains("_old", case=False, na=False)]
    print(f"\n  dopo il filtro grossolano: {len(df)} ticker (da {before})")
    print(f"    di cui delistati: {int(df['is_delisted'].sum())}  ← eliminano il survivorship bias")

    if args.limit and len(df) > args.limit:
        # Campionamento RIPRODUCIBILE. `df.sample(random_state=42)` sembra
        # deterministico ma non lo è: campiona POSIZIONI, quindi se EODHD
        # aggiunge o toglie un solo ticker dalla lista, l'intero campione
        # cambia. (Successo davvero: 20899 -> 20898 ticker, e il 54% del
        # campione era diverso, rendendo inutile la cache già scaricata.)
        #
        # Con l'hash del TICKER ogni titolo ha un destino fisso: chi entrava
        # ieri entra oggi, indipendentemente da chi altro c'è nella lista.
        h = df["ticker"].map(
            lambda t: int(hashlib.md5(t.encode()).hexdigest()[:8], 16))  # noqa: S324
        df = df.assign(_h=h).nsmallest(args.limit, "_h").drop(columns="_h")
        df = df.sort_values("ticker")
        print(f"\n⚠ limito a {args.limit} ticker (--limit).")
        print("  Selezione deterministica per hash del ticker: stabile anche se")
        print("  la lista di EODHD cambia. NON è l'ordine alfabetico, che")
        print("  concentrerebbe i delistati anziani e falserebbe la prova.")
        print("  Resta una selezione arbitraria: per il backtest vero togli --limit.")

    out = Path(args.data_dir) / f"universe_{args.exchange}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Si conservano i METADATI, non solo il ticker. Senza `type`, `exchange_real`
    # e `currency` non si può verificare cosa contenga l'universo, né capire a
    # posteriori perché un titolo sia (o non sia) nel campione.
    keep = [c for c in ("ticker", "is_delisted", "type", "exchange_real",
                        "currency", "name") if c in df.columns]
    df[keep].to_parquet(out)
    print(f"\n✓ universo salvato in {out}: {len(df)} ticker")
    if "exchange_real" in df.columns:
        top = df["exchange_real"].value_counts().head(6)
        print("  composizione per borsa: " +
              ", ".join(f"{k} {v:,}" for k, v in top.items()))
    print(f"\nProssimo passo (attenzione: {len(df)} chiamate API):")
    print(f"  python build_universe.py prices --from-universe {out} --start {args.start}")


def cmd_current(args) -> None:
    """Composizione ATTUALE di un indice: l'unica disponibile per l'Europa.
    Genera un frammento YAML da incollare nel config, col bias dichiarato."""
    provider = make_provider("eodhd")
    df = provider.current_constituents(args.index)
    if df.empty:
        print(f"Nessun componente per {args.index}"); return
    print(f"✓ {args.index}: {len(df)} componenti ATTUALI\n")
    print("⚠ ATTENZIONE: questa è la fotografia di OGGI. Usarla su 20 anni di")
    print("  storia reintroduce il survivorship bias: stai selezionando aziende")
    print("  che sappiamo essere sopravvissute. I risultati vanno letti come")
    print("  LIMITE SUPERIORE della performance reale.\n")
    out = Path(args.data_dir) / f"current_{args.index.replace('.','_')}.yaml"
    lines = ["    tickers:"] + [f"      - {t}" for t in sorted(df["ticker"])]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Lista salvata in {out} — incollala sotto `universe:` nel config.")


def cmd_info(args) -> None:
    """Ispeziona un universo già costruito: composizione, borse, tipi.
    Serve a verificare COSA c'è dentro prima di scaricare ore di storici."""
    up = Path(args.data_dir) / f"universe_{args.exchange}.parquet"
    if up.exists():
        df = pd.read_parquet(up)
        print(f"Universo {up}: {len(df):,} ticker")
        print(f"  delistati: {int(df['is_delisted'].sum()):,} "
              f"({df['is_delisted'].mean() * 100:.0f}%)  ← eliminano il survivorship bias")
        for col, label in (("type", "tipo"), ("exchange_real", "borsa"),
                           ("currency", "valuta")):
            if col in df.columns:
                vc = df[col].value_counts().head(8)
                print(f"\n  per {label}:")
                for k, v in vc.items():
                    print(f"    {str(k):20s} {v:>8,}")
        if "exchange_real" not in df.columns:
            print("\n  ⚠ metadati assenti: universo costruito con una versione")
            print("    precedente. Rilancia `screen` per averli.")
        return

    # altrimenti: universo point-in-time da indice
    uni = PointInTimeUniverse.load(args.data_dir, args.index)
    s = uni.stats()
    print(json.dumps(s, indent=2, ensure_ascii=False))
    for d in (args.start, "2008-09-01", "2020-03-01", args.end):
        try:
            print(f"  membri al {d}: {len(uni.active_at(d))}")
        except Exception:  # noqa: BLE001, S110
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Universo storico + cache prezzi")
    ap.add_argument("command", choices=["screen", "constituents", "current", "prices", "info"])
    ap.add_argument("--exchange", default="US", help="borsa per `screen` (US, LSE, XETRA...)")
    ap.add_argument("--from-universe", default=None, help="parquet prodotto da `screen`")
    ap.add_argument("--with-sector-etfs", action="store_true",
                    help="scarica anche gli 11 ETF settoriali SPDR")
    # `--exclude-otc` è attivo di DEFAULT: l'OTC ospita shell company e penny
    # stock che il filtro di liquidità scarterebbe comunque, e scaricarne lo
    # storico costa ore. `--include-otc` lo riattiva.
    # (Con `action="store_true", default=True` il flag era inutilizzabile:
    #  restava sempre True qualunque cosa si passasse.)
    ap.add_argument("--include-otc", dest="exclude_otc", action="store_false",
                    default=True, help="includi i titoli OTC (default: esclusi)")
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--limit", type=int, default=None,
                    help="SOLO PER PROVE. Campiona N ticker per hash (deterministico). "
                         "Non selezionare 'i più liquidi': la liquidità di oggi è "
                         "informazione futura e riporterebbe il survivorship bias.")
    ap.add_argument("--index", default="GSPC.INDX", help="es. GSPC.INDX (S&P 500)")
    ap.add_argument("--benchmark", default=None, help="default: l'indice stesso")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--yes", action="store_true", help="non chiedere conferma")
    args = ap.parse_args()

    {"screen": cmd_screen, "constituents": cmd_constituents, "current": cmd_current,
     "prices": cmd_prices, "info": cmd_info}[args.command](args)


if __name__ == "__main__":
    main()
