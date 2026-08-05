#!/usr/bin/env python3
"""Scrive i metadati per la cache già scaricata (una tantum).

Perché serve. La cache di un ticker si considera valida se copre il periodo
richiesto, warmup incluso. Ma il warmup risale 80 settimane prima dello
`start` del config: con `start: 2001-06-01` si arriva al 1999-11-19, mentre
i dati sono stati scaricati dal 2000-01-01. La cache sembrerebbe insufficiente
e verrebbe RISCARICATA per intero — ore di chiamate API.

Da questa versione registriamo in un `.meta.json` da quale data i dati sono
stati CHIESTI: se abbiamo già chiesto da lì, quel che c'è è tutto quel che il
provider aveva. Questo script scrive i metadati per i file già presenti.

  python migrate_cache.py --start 2000-01-01
"""
from __future__ import annotations

import sys
# UTF-8 su Windows: la console usa cp1252 e crasha sui simboli (→ × ✓ ⚠).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--provider", default="eodhd")
    ap.add_argument("--start", required=True,
                    help="la data --start usata in `build_universe.py prices` "
                         "(es. 2000-01-01)")
    ap.add_argument("--warmup-weeks", type=int, default=80,
                    help="deve coincidere col default di get_with_warmup")
    args = ap.parse_args()

    # `prices --start 2000-01-01` NON chiede al provider dal 2000-01-01:
    # get_with_warmup sottrae 80 settimane e chiede dal 1999-11-19. È QUELLA
    # la data da registrare, altrimenti la cache resta invalida e si riscarica.
    asked_from = (pd.Timestamp(args.start)
                  - pd.Timedelta(weeks=args.warmup_weeks)).strftime("%Y-%m-%d")
    print(f"--start {args.start} + warmup {args.warmup_weeks}w → "
          f"i dati furono chiesti dal {asked_from}")

    root = Path(args.data_dir) / args.provider
    if not root.exists():
        print(f"✗ cache non trovata: {root}")
        raise SystemExit(1)

    files = sorted(root.glob("*.parquet"))
    written = skipped = 0
    for p in files:
        meta = p.with_suffix(".meta.json")
        if meta.exists():
            skipped += 1
            continue
        try:
            df = pd.read_parquet(p)
            asked_to = str(df.index.max().date()) if len(df) else args.start
        except Exception:  # noqa: BLE001
            asked_to = args.start
        meta.write_text(json.dumps({"asked_from": asked_from, "asked_to": asked_to}))
        written += 1

    print(f"✓ metadati scritti: {written} · già presenti: {skipped}")
    print(f"  cache: {len(files)} ticker in {root}")
    print("\n  Ora il backtest userà la cache invece di riscaricare tutto.")


if __name__ == "__main__":
    main()
