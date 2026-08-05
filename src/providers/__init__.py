"""Registry dei provider: il resto del codice non sa quale fonte sta usando."""
from __future__ import annotations

from .base import DataProvider
from .synthetic import SyntheticProvider


def make_provider(name: str) -> DataProvider:
    if name == "synthetic":
        return SyntheticProvider()
    if name == "eodhd":
        from .eodhd import EODHDProvider  # import pigro: requests/chiave solo se serve
        return EODHDProvider()
    raise ValueError(f"Provider sconosciuto: {name!r} (validi: synthetic, eodhd)")
