"""Universo point-in-time: chi era nell'indice a ogni data.

Il punto di tutto questo modulo è una sola frase: **il backtest può comprare
solo titoli che a quella data erano effettivamente nell'indice**.

Perché è cruciale. Se prendi i 500 componenti ATTUALI dell'S&P 500 e li testi
sul 2000-2025, stai testando aziende che sappiamo essere sopravvissute e
cresciute abbastanza da restare (o entrare) nell'indice. Enron, Lehman,
Wirecard e centinaia di nomi minori spariscono dal campione. Per una strategia
di breakout questo gonfia i risultati, perché molti falsi breakout storici
sono avvenuti proprio su titoli poi morti. È il "survivorship bias", ed è il
singolo bias più distorsivo per questo tipo di strategia.

La struttura dati di EODHD (ticker, start, end, is_delisted) permette di
ricostruire l'appartenenza a ogni settimana. Un titolo delistato nel 2008
resta comprabile fino al 2008 e poi sparisce — esattamente come nella realtà.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CONSTITUENTS_FILE = "constituents_{index}.parquet"


@dataclass
class PointInTimeUniverse:
    """Appartenenza all'indice nel tempo. `members` ha colonne:
    ticker, start, end (NaT = ancora dentro), is_delisted."""

    members: pd.DataFrame
    index_name: str

    # ------------------------------------------------------------------
    def active_at(self, date: pd.Timestamp) -> list[str]:
        """I ticker che a questa data erano nell'indice."""
        d = pd.Timestamp(date)
        m = self.members
        started = m["start"].isna() | (m["start"] <= d)
        not_ended = m["end"].isna() | (m["end"] > d)
        return sorted(m.loc[started & not_ended, "ticker"].tolist())

    def all_tickers(self) -> list[str]:
        """Tutti i ticker mai apparsi nel periodo: è l'insieme da scaricare."""
        return sorted(self.members["ticker"].unique().tolist())

    def membership_matrix(self, weeks: pd.DatetimeIndex) -> pd.DataFrame:
        """Matrice booleana settimane × ticker: True se il titolo era
        nell'indice quella settimana. È la struttura che il backtest consulta
        prima di aprire una posizione."""
        tickers = self.all_tickers()
        mat = pd.DataFrame(False, index=weeks, columns=tickers)
        for _, row in self.members.iterrows():
            start = row["start"] if pd.notna(row["start"]) else weeks[0]
            end = row["end"] if pd.notna(row["end"]) else weeks[-1] + pd.Timedelta(days=1)
            mask = (weeks >= start) & (weeks < end)
            mat.loc[mask, row["ticker"]] = True
        return mat

    def stats(self) -> dict:
        m = self.members
        return {
            "index": self.index_name,
            "tickers_totali": len(m),
            "delistati": int(m["is_delisted"].sum()),
            "ancora_attivi": int(m["end"].isna().sum()),
            "primo_ingresso": str(m["start"].min().date()) if m["start"].notna().any() else None,
            "ultima_uscita": str(m["end"].max().date()) if m["end"].notna().any() else None,
        }

    # ------------------------------------------------------------------
    def save(self, data_dir: str | Path) -> Path:
        p = Path(data_dir) / CONSTITUENTS_FILE.format(index=self.index_name.replace(".", "_"))
        p.parent.mkdir(parents=True, exist_ok=True)
        self.members.to_parquet(p)
        return p

    @classmethod
    def load(cls, data_dir: str | Path, index_name: str) -> "PointInTimeUniverse":
        p = Path(data_dir) / CONSTITUENTS_FILE.format(index=index_name.replace(".", "_"))
        if not p.exists():
            raise FileNotFoundError(
                f"Composizione storica non in cache: {p}\n"
                f"  Esegui prima:  python build_universe.py --index {index_name}"
            )
        return cls(pd.read_parquet(p), index_name)

    @classmethod
    def from_provider(cls, provider, index_name: str) -> "PointInTimeUniverse":
        df = provider.historical_constituents(index_name)
        return cls(df, index_name)

    @classmethod
    def static(cls, tickers: list[str], index_name: str = "static") -> "PointInTimeUniverse":
        """Universo senza storia: tutti i titoli presenti per tutto il periodo.
        ATTENZIONE: reintroduce il survivorship bias. Da usare solo quando la
        composizione storica non è disponibile (es. indici europei), e da
        DICHIARARE nel report."""
        df = pd.DataFrame({"ticker": tickers, "start": pd.NaT, "end": pd.NaT,
                           "is_delisted": False, "is_active": True})
        return cls(df, index_name)


class LiquidityUniverse:
    """Universo definito da una REGOLA sui dati, non dall'appartenenza a un indice.

    Perché è metodologicamente superiore all'indice per il metodo Weinstein:

    1. **Point-in-time gratis.** Prezzo e volume di quella settimana sono già
       nei dati EOD. Un titolo liquido nel 2005 entra nell'universo del 2005;
       se muore nel 2008, esce nel 2008. Nessun senno di poi, nessun bisogno
       di dati fondamentali a pagamento.

    2. **Cattura i titoli giusti.** L'S&P 500 contiene solo mega-cap, che
       passano anni in Fase 2 senza formare basi. Weinstein cerca chi ESCE da
       una base: le mid-cap liquide sono il terreno naturale del metodo.

    3. **Replicabile.** La composizione di un indice è la decisione di un
       comitato; una soglia di liquidità è una regola che chiunque riproduce.

    Sui filtri: la capitalizzazione richiederebbe il numero di azioni, che sta
    nei Fundamentals. Usiamo il **turnover in dollari** (prezzo × volume) come
    proxy della dimensione — e per uno screener è persino più pertinente,
    perché ciò che conta è poter entrare e uscire senza muovere il prezzo.
    """

    def __init__(self, min_price: float = 5.0, min_volume: float = 200_000,
                 min_dollar_volume: float = 5_000_000, lookback_weeks: int = 13,
                 max_volatility: float | None = None, atr_weeks: int = 14) -> None:
        self.min_price = min_price
        self.min_volume = min_volume
        self.min_dollar_volume = min_dollar_volume
        self.lookback_weeks = lookback_weeks
        # Volatilità massima: ATR normalizzato sul prezzo (ATR% settimanale).
        # Weinstein preferisce società di una certa dimensione perché danno
        # "liquidità buona e variazioni contenute". Ma la capitalizzazione è
        # il PROXY, non l'obiettivo: la liquidità la misura già il turnover,
        # e le "variazioni contenute" sono volatilità — misurabile DIRETTAMENTE
        # dai prezzi, senza dati fondamentali a pagamento e senza il rischio di
        # usare la cap odierna sulla storia passata (che sarebbe look-ahead:
        # selezionerebbe le small-cap del 2005 diventate mega-cap oggi).
        # None = criterio disattivato.
        self.max_volatility = max_volatility
        self.atr_weeks = atr_weeks
        self._eligible: dict[str, pd.Series] = {}   # ticker -> booleana per settimana
        self._diag: dict[str, pd.DataFrame] = {}    # ticker -> turnover/volatilità

    def compute(self, ticker: str, weekly: pd.DataFrame) -> pd.Series:
        """Idoneità settimana per settimana. Usa medie sulle settimane
        PRECEDENTI (shift): alla settimana t non si conosce il volume di t
        quando si decide se t è nell'universo. Anti look-ahead."""
        from .indicators import atr as _atr

        px = weekly["adj_close"]
        vol = weekly["volume"]
        dollar = px * vol
        n = self.lookback_weeks
        # shift(1) su TUTTI i criteri, prezzo incluso: alla settimana t
        # l'idoneità si giudica su ciò che era noto a t-1. Senza lo shift sul
        # prezzo, un titolo che sfonda i $5 proprio nella settimana del
        # breakout diventerebbe idoneo grazie al breakout stesso.
        prev_px = px.shift(1)
        avg_vol = vol.shift(1).rolling(n, min_periods=n).mean()
        avg_dollar = dollar.shift(1).rolling(n, min_periods=n).mean()
        ok = (
            (prev_px >= self.min_price)
            & (avg_vol >= self.min_volume)
            & (avg_dollar >= self.min_dollar_volume)
        )

        # ATR% = volatilità normalizzata. Anche qui shift(1): la barra del
        # breakout è per definizione ampia, e senza shift escluderebbe da sola
        # i titoli proprio nel momento in cui danno il segnale.
        atr_pct = (_atr(weekly, self.atr_weeks) / px).shift(1)
        if self.max_volatility is not None:
            ok &= atr_pct <= self.max_volatility

        ok = ok.fillna(False)
        self._eligible[ticker] = ok
        # conservati per la diagnostica: il turnover alto implica già bassa
        # volatilità? Se sì, il criterio è ridondante e i numeri lo diranno.
        self._diag[ticker] = pd.DataFrame({"turnover": avg_dollar, "atr_pct": atr_pct})
        return ok

    def redundancy_check(self) -> dict:
        """Il filtro di volatilità aggiunge qualcosa, o il turnover lo implica già?

        Domanda empirica, non retorica. Se i titoli ad alto turnover fossero
        sempre a bassa volatilità, il criterio sarebbe ridondante e andrebbe
        tolto (ogni filtro in più riduce i segnali e aumenta l'overfitting).
        Mettiamo i numeri, invece di assumere.
        """
        rows = []
        for tk, d in self._diag.items():
            d = d.dropna()
            if len(d) < 20:
                continue
            rows.append(d.assign(ticker=tk))
        if not rows:
            return {"disponibile": False}
        allv = pd.concat(rows)
        # se turnover o volatilità sono costanti, la correlazione non è definita:
        # meglio dirlo che stampare 'nan'
        if allv["turnover"].nunique() < 2 or allv["atr_pct"].nunique() < 2:
            corr = None
        else:
            # Spearman = Pearson sui RANGHI. Lo calcoliamo così invece di usare
            # method="spearman", che in pandas richiede scipy (~40 MB) per una
            # sola correlazione. rank() + corr() sono nativi.
            corr = allv["turnover"].rank().corr(allv["atr_pct"].rank())
            corr = None if pd.isna(corr) else round(float(corr), 3)

        out = {"disponibile": True,
               "correlazione_turnover_volatilita": corr,
               "atr_pct_mediano": round(float(allv["atr_pct"].median()), 4)}
        if self.max_volatility is not None:
            # quanti titoli passerebbero il turnover ma NON la volatilità?
            passa_turnover = allv["turnover"] >= self.min_dollar_volume
            fallisce_vol = allv["atr_pct"] > self.max_volatility
            solo_vol = int((passa_turnover & fallisce_vol).sum())
            out["settimane_escluse_solo_da_volatilita"] = solo_vol
            out["quota_esclusa_solo_da_volatilita_pct"] = round(
                solo_vol / max(int(passa_turnover.sum()), 1) * 100, 1)
            out["interpretazione"] = (
                "il filtro volatilità aggiunge selettività" if solo_vol > 0
                else "RIDONDANTE: il turnover implica già bassa volatilità")
        return out

    def is_eligible(self, ticker: str, when: pd.Timestamp) -> bool:
        s = self._eligible.get(ticker)
        if s is None:
            return False
        if when in s.index:
            return bool(s.loc[when])
        idx = s.index.asof(when)
        return bool(s.loc[idx]) if idx is not pd.NaT and not pd.isna(idx) else False

    def membership_dict(self, ticker: str) -> dict:
        s = self._eligible.get(ticker)
        return s.to_dict() if s is not None else {}

    def stats(self) -> dict:
        if not self._eligible:
            return {"tickers": 0}
        mat = pd.DataFrame(self._eligible)
        per_week = mat.sum(axis=1)
        return {
            "tickers_valutati": len(self._eligible),
            "mai_idonei": int((mat.sum(axis=0) == 0).sum()),
            "membri_per_settimana_min": int(per_week.min()),
            "membri_per_settimana_medio": int(per_week.mean()),
            "membri_per_settimana_max": int(per_week.max()),
            "soglie": {"prezzo_min": self.min_price, "volume_min": self.min_volume,
                       "turnover_min": self.min_dollar_volume,
                       "volatilita_max": self.max_volatility},
        }
