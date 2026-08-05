#!/usr/bin/env python3
"""Verifica della connessione EODHD prima di lanciare fetch/backtest.

Uso:
  export EODHD_API_KEY=demo          # o la tua chiave gratuita
  python check_eodhd.py

Controlla, spendendo pochissime chiamate:
  1) che la chiave sia raggiungibile e quanta quota resta (User API);
  2) un fetch di prova su un ticker, mostrando prime/ultime righe e
     segnalando se lo storico è troppo corto per la MA30 (caso della
     chiave gratuita, limitata all'ultimo anno).
"""
from __future__ import annotations

import os
import sys

# UTF-8 su Windows: la console usa cp1252 e crasha sui simboli (→ × ✓ ⚠).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


import requests

API_ROOT = "https://eodhd.com/api"
# ticker pieni sotto chiave demo; con chiave propria si può cambiare
PROBE = "AAPL.US"


def main() -> int:
    key = os.environ.get("EODHD_API_KEY", "")
    if not key:
        print("✗ EODHD_API_KEY non impostata.\n"
              "  Esegui:  export EODHD_API_KEY=demo   (oppure la tua chiave)")
        return 1
    shown = key if key == "demo" else key[:4] + "…" + key[-2:]
    print(f"Chiave in uso: {shown}\n")

    # 1) stato account / quota
    try:
        u = requests.get(f"{API_ROOT}/user", params={"api_token": key, "fmt": "json"},
                         timeout=20)
        u.raise_for_status()
        info = u.json()
        used = info.get("apiRequests", "?")
        limit = info.get("dailyRateLimit", "?")
        print(f"✓ Connessione OK · chiamate usate oggi: {used}/{limit}")
    except Exception as e:  # noqa: BLE001
        print(f"✗ User API fallita: {e}")
        print("  Se la chiave è 'demo', la User API può non rispondere: proseguo col fetch.")

    # 2) fetch di prova
    try:
        r = requests.get(f"{API_ROOT}/eod/{PROBE}",
                         params={"api_token": key, "fmt": "json", "period": "d",
                                 "from": "2000-01-01", "to": "2024-12-31"},
                         timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            print(f"✗ Nessun dato per {PROBE}.")
            return 1
        first, last = rows[0]["date"], rows[-1]["date"]
        span_days = _days_between(first, last)
        print(f"\n✓ {PROBE}: {len(rows)} barre  ({first} → {last})")
        print(f"   prima riga: {rows[0]}")
        print(f"   ultima riga: {rows[-1]}")
        if span_days < 500:
            print("\n⚠ Storico < ~1,5 anni: probabilmente sei sulla chiave GRATUITA,\n"
                  "  che restituisce solo l'ultimo anno. La MA a 30 settimane non\n"
                  "  può maturare. Per un test reale usa la chiave 'demo' (storico\n"
                  "  completo su AAPL/TSLA/AMZN/VTI) o un piano a pagamento.")
        else:
            print("\n✓ Storico sufficiente per la MA30: puoi lanciare il backtest.\n"
                  "  python run.py backtest --config config_demo_eodhd.yaml")
    except requests.HTTPError as e:
        print(f"✗ Fetch fallito: {e}\n  Verifica la chiave o il formato del ticker.")
        return 1
    return 0


def _days_between(d1: str, d2: str) -> int:
    import datetime as dt
    a = dt.date.fromisoformat(d1)
    b = dt.date.fromisoformat(d2)
    return (b - a).days


if __name__ == "__main__":
    sys.exit(main())
