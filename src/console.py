"""Console UTF-8 anche su Windows.

La console di Windows usa cp1252 e solleva UnicodeEncodeError sui caratteri
Unicode che il progetto stampa spesso: frecce (→), moltiplicazione (×), spunte
(✓), avvisi (⚠). Senza riconfigurazione, il programma crasha alla prima print
con uno di questi simboli — cosa che è successa davvero, fermando un backtest
al primo output.

Importare `from src.console import setup` e chiamarlo in cima a ogni entry point.
"""
from __future__ import annotations

import sys


def setup() -> None:
    """Riconfigura stdout/stderr in UTF-8. Idempotente, silenzioso se non serve."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            # stream non riconfigurabile (redirect, ambiente particolare): pazienza,
            # meglio proseguire che crashare qui.
            pass
