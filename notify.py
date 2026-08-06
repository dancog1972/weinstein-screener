#!/usr/bin/env python3
"""Notifica Telegram: a fine run manda SEMPRE un "Weekly Summary" con lo stato
(nuovi segnali pieni o "nessuno"), il recap del follow-up (pieni totali + quasi
seguiti) e i link a screener e follow-up.

Env (da GitHub Secrets): TELEGRAM_TOKEN, TELEGRAM_CHAT_ID · SITE_URL (link) ·
WATCH_URL/WATCH_SECRET (per contare i quasi seguiti, opzionale).
Se TELEGRAM_TOKEN/CHAT_ID non ci sono → non fa nulla.
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


def _watchlist_count() -> int | None:
    """Quanti 'quasi' segui (dal Worker Cloudflare). None se non configurato."""
    url, secret = os.environ.get("WATCH_URL"), os.environ.get("WATCH_SECRET")
    if not url or not secret:
        return None
    try:
        r = requests.get(f"{url.rstrip('/')}/list", params={"secret": secret}, timeout=15)
        r.raise_for_status()
        return len(r.json() or [])
    except Exception:  # noqa: BLE001
        return None


def _subscribers() -> list[str]:
    """chat_id degli iscritti al recap (dal Worker). Vuoto se non configurato."""
    url, secret = os.environ.get("WATCH_URL"), os.environ.get("WATCH_SECRET")
    if not url or not secret:
        return []
    try:
        r = requests.get(f"{url.rstrip('/')}/subscribers", params={"secret": secret}, timeout=15)
        r.raise_for_status()
        return [str(s["chat_id"]) for s in (r.json() or []) if s.get("chat_id")]
    except Exception:  # noqa: BLE001
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Recap settimanale su Telegram")
    ap.add_argument("--log", default="signals_log.json")
    args = ap.parse_args()

    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("  [telegram] TELEGRAM_TOKEN/CHAT_ID non impostati — salto")
        return

    logp = Path(args.log)
    log = json.loads(logp.read_text(encoding="utf-8")) if logp.exists() else []
    today = date.today().isoformat()
    new = [e for e in log if e.get("first_logged") == today]   # pieni aggiunti in QUESTO run
    n_quasi = _watchlist_count()

    lines = [f"📊 <b>Weekly Summary</b> — {today}"]
    if new:
        lines.append(f"🟢 <b>{len(new)} nuovi segnali PIENI</b>:")
        for e in new[:15]:
            lines.append(f"• <b>{e['ticker']}</b> ({e.get('market','')}) · entry {e.get('entry')} · "
                         f"stop {e.get('stop')} · Mans {e.get('mansfield')}")
    else:
        lines.append("⚪️ Nessun nuovo segnale.")

    recap = f"Follow-up: <b>{len(log)}</b> pieni totali"
    if n_quasi is not None:
        recap += f" · <b>{n_quasi}</b> quasi seguiti"
    lines.append(recap)

    site = os.environ.get("SITE_URL", "").strip().rstrip("/")
    if site:
        lines.append(f'🔎 <a href="{site}/">Screener</a> · '
                     f'📋 <a href="{site}/signals.html">Follow-up</a> · '
                     f'📖 <a href="{site}/metodo.html">Metodo</a>')

    text = "\n".join(lines)
    recipients = {str(chat)} | set(_subscribers())      # tu + gli iscritti (dedup)
    ok = 0
    for cid in recipients:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": cid, "text": text, "parse_mode": "HTML",
                      "disable_web_page_preview": True}, timeout=20)
            if r.ok:
                ok += 1
            else:
                print(f"  [telegram] {cid}: errore {r.status_code} {r.text[:120]}")
        except Exception as e:  # noqa: BLE001
            print(f"  [telegram] {cid}: invio fallito {e}")
    print(f"  [telegram] recap inviato a {ok}/{len(recipients)} destinatari")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
