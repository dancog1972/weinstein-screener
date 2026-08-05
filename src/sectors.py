"""Settori: il secondo schermo del triple screen di Weinstein.

Weinstein insiste su una gerarchia: **mercato → settore → titolo**
("forest before trees"). Comprare il miglior titolo di un settore debole è
sbagliato quanto comprare un titolo debole in un settore forte.

Come misuriamo la forza di un settore senza dati fondamentali:
usiamo gli **ETF settoriali SPDR** come rappresentanti. La loro serie storica
è scaricabile col piano EOD base, e possiamo calcolarne fase e Mansfield RS
con lo stesso motore usato per i titoli. Il settore è "forte" quando il suo
ETF è in Fase 2 con RS positiva.

Due limiti, dichiarati:
1. La mappa ticker→settore è STATICA (classificazione di oggi applicata alla
   storia). Le classificazioni cambiano: Amazon è passata da Retail a
   Consumer Discretionary. È un bias di classificazione, molto più lieve del
   survivorship bias, ma esiste.
2. Gli ETF SPDR esistono dal dicembre 1998; XLC (Communication Services) solo
   dal 2018. Prima di quelle date il filtro settoriale non si applica e il
   codice lo dichiara invece di inventare dati.
"""
from __future__ import annotations

import pandas as pd

# Gli 11 settori GICS e i loro ETF SPDR rappresentativi.
# XLC è nato nel 2018 dallo scorporo di Telecom: prima di allora quei titoli
# stavano in Technology (XLK) e Consumer Discretionary (XLY).
SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK.US",
    "Financial Services": "XLF.US",
    "Healthcare": "XLV.US",
    "Consumer Cyclical": "XLY.US",
    "Consumer Defensive": "XLP.US",
    "Industrials": "XLI.US",
    "Energy": "XLE.US",
    "Basic Materials": "XLB.US",
    "Utilities": "XLU.US",
    "Real Estate": "XLRE.US",          # dal 2015
    "Communication Services": "XLC.US",  # dal 2018
}

# Da quando ogni ETF ha storia utilizzabile (prima: filtro settoriale inattivo)
ETF_INCEPTION: dict[str, str] = {
    "XLK.US": "1998-12-22", "XLF.US": "1998-12-22", "XLV.US": "1998-12-22",
    "XLY.US": "1998-12-22", "XLP.US": "1998-12-22", "XLI.US": "1998-12-22",
    "XLE.US": "1998-12-22", "XLB.US": "1998-12-22", "XLU.US": "1998-12-22",
    "XLRE.US": "2015-10-08", "XLC.US": "2018-06-19",
}

# Fallback: quando l'ETF specifico non esiste ancora a quella data,
# usiamo il settore che allora conteneva quei titoli.
ETF_PREDECESSOR: dict[str, str] = {
    "XLRE.US": "XLF.US",   # Real Estate stava nei Financials fino al 2015
    "XLC.US": "XLK.US",    # Communication Services stava in Technology
}


class SectorMap:
    """Mappa ticker → settore → ETF proxy, con la fase del settore nel tempo."""

    def __init__(self, ticker_sector: dict[str, str]) -> None:
        self.ticker_sector = ticker_sector
        self._stage: dict[str, pd.Series] = {}    # etf -> fase settimanale
        self._rs: dict[str, pd.Series] = {}       # etf -> Mansfield RS

    # ------------------------------------------------------------------
    def sector_of(self, ticker: str) -> str | None:
        return self.ticker_sector.get(ticker)

    def etf_for(self, ticker: str, when: pd.Timestamp | None = None) -> str | None:
        """L'ETF che rappresenta il settore del titolo a quella data.
        Se l'ETF non esisteva ancora, ripiega sul predecessore."""
        sector = self.sector_of(ticker)
        if sector is None:
            return None
        etf = SECTOR_ETFS.get(sector)
        if etf is None or when is None:
            return etf
        inception = pd.Timestamp(ETF_INCEPTION.get(etf, "1900-01-01"))
        if when < inception:
            return ETF_PREDECESSOR.get(etf)     # None se non c'è predecessore
        return etf

    def register_sector_series(self, etf: str, stage: pd.Series, rs: pd.Series) -> None:
        self._stage[etf] = stage
        self._rs[etf] = rs

    def etfs_needed(self) -> list[str]:
        """Gli ETF da scaricare per coprire i settori presenti nell'universo.

        Include i PREDECESSORI: XLRE nasce nel 2015, ma prima di allora il
        Real Estate stava dentro XLF. Senza il predecessore, tutta la storia
        pre-2015 di quel settore resterebbe scoperta e il filtro inattivo.
        """
        sectors = set(self.ticker_sector.values())
        etfs = {SECTOR_ETFS[s] for s in sectors if s in SECTOR_ETFS}
        etfs |= {ETF_PREDECESSOR[e] for e in list(etfs) if e in ETF_PREDECESSOR}
        return sorted(etfs)

    # ------------------------------------------------------------------
    def sector_ok(self, ticker: str, when: pd.Timestamp,
                  require_stage2: bool = True, require_rs_positive: bool = True) -> bool:
        """Il secondo schermo: il settore del titolo è forte a questa data?

        Se il settore è ignoto o l'ETF non ha ancora storia, ritorna True:
        il filtro NON si applica invece di scartare arbitrariamente. Meglio
        un filtro assente e dichiarato che un filtro inventato.
        """
        etf = self.etf_for(ticker, when)
        if etf is None or etf not in self._stage:
            return True
        st = self._stage[etf]
        idx = st.index.asof(when)
        if idx is pd.NaT or pd.isna(idx):
            return True
        if require_stage2 and int(st.loc[idx]) != 2:
            return False
        if require_rs_positive and etf in self._rs:
            rs = self._rs[etf]
            r = rs.loc[idx] if idx in rs.index else None
            if r is not None and not pd.isna(r) and r < 0:
                return False
        return True

    def coverage(self) -> dict:
        """Diagnostica della mappa. `settori_sconosciuti` è la voce da guardare:
        un nome di settore che non corrisponde a nessun ETF disattiva il filtro
        per quei titoli SILENZIOSAMENTE. La nomenclatura attesa è quella di
        EODHD/Morningstar ('Financial Services', non 'Financials')."""
        known = sum(1 for s in self.ticker_sector.values() if s in SECTOR_ETFS)
        unknown = sorted({s for s in self.ticker_sector.values() if s not in SECTOR_ETFS})
        return {"tickers_mappati": len(self.ticker_sector),
                "con_settore_noto": known,
                "settori_sconosciuti": unknown,
                "etf_caricati": sorted(self._stage.keys())}

    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, path: str) -> "SectorMap":
        """CSV con colonne: ticker,sector"""
        df = pd.read_csv(path)
        return cls(dict(zip(df["ticker"], df["sector"], strict=False)))

    @classmethod
    def empty(cls) -> "SectorMap":
        """Nessuna mappa: il filtro settoriale è inattivo (dichiarato)."""
        return cls({})
