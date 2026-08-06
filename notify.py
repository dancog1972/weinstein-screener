#!/usr/bin/env python3
"""Notifica Telegram: a fine run, se sono comparsi NUOVI segnali PIENI (aggiunti
al log in questa esecuzione), manda la lista + il link alla pagina. Solo i nuovi,
così non ti arriva lo stesso segnale ogni settimana.

Env (da GitHub Secrets): TELEGRAM_TOKEN, TELEGRAM_CHAT_ID · SITE_URL (link).
Se i secret non ci sono → non fa nulla (degrada in silenzio).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

from src.console import setup as _console_setup

_console_setup()

import requests


def main() -> None:
    ap = argparse.ArgumentParser(description="Notifica Telegram dei nuovi segnali pieni")
    ap.add_argument("--log", default="signals_log.json")
    args = ap.parse_args()

    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("  [telegram] TELEGRAM_TOKEN/CHAT_ID non impostati — salto")
        return
    logp = Path(args.log)
    if not logp.exists():
        print("  [telegram] signals_log.json assente — salto")
        return

    log = json.loads(logp.read_text(encoding="utf-8"))
    today = date.today().isoformat()
    new = [e for e in log if e.get("first_logged") == today]   # aggiunti in QUESTO run
    if not new:
        print("  [telegram] nessun nuovo segnale pieno — nessuna notifica")
        return

    site = os.environ.get("SITE_URL", "").strip()
    lines = [f"🟢 <b>Screener Weinstein</b> — {len(new)} nuovo/i segnale/i PIENO/I"]
    for e in new[:15]:
        lines.append(f"• <b>{e['ticker']}</b> ({e.get('market','')}) · entry {e.get('entry')} · "
                     f"stop {e.get('stop')} · Mansfield {e.get('mansfield')}")
    if site:
        lines.append(f'👉 <a href="{site}">apri lo screener</a>')
    text = "\n".join(lines)

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True}, timeout=20)
        print("  [telegram] notifica inviata ✓" if r.ok
              else f"  [telegram] errore {r.status_code}: {r.text[:200]}")
    except Exception as e:  # noqa: BLE001
        print(f"  [telegram] invio fallito: {e}")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
