"""Motore del simulatore human-vs-bot.

Idea centrale: una SESSIONE avanza settimana per settimana ma si FERMA solo
sugli eventi (nuovo segnale, o stop/breakdown su una posizione aperta). A
ogni fermata, umano e bot vedono lo stesso stato e decidono in parallelo, con
DUE portafogli separati ma regole di contorno identiche. Alla fine si
confrontano le due curve equity contro il benchmark.

Riusa il motore esistente: le fasi, i segnali e il portafoglio sono gli stessi
del backtest automatico. Qui cambia solo CHI decide sull'ingresso: il bot
applica le regole, l'umano sceglie via API.

Anti-hindsight: in modalità 'blind' ticker e date reali sono sostituiti da
alias stabili ("STOCK-03") e da un contatore di settimane. La mappa di
de-anonimizzazione resta lato server e non viene mai inviata al client finché
la sessione non è chiusa.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import numpy as np
import pandas as pd

from .backtest import load_universe
from .config import load_config
from .data_store import DataStore
from .indicators import (confirmed_pivot_lows, enrich_weekly,
                         last_confirmed_pivot_before, to_weekly)
from .portfolio import Portfolio
from .providers import make_provider
from .signals import Signal, detect_signals, prepare_ticker
from .stages import STAGE_NAMES, classify_stages

Actor = Literal["human", "bot"]


def _py(obj):
    """Converte ricorsivamente i tipi numpy/pandas in tipi Python nativi.

    Necessario perché numpy.int64 e numpy.float64 non sono serializzabili in
    JSON: senza questa conversione l'export fallisce in modo silenzioso e
    imprevedibile (dipende da quali valori capitano interi).
    """
    if isinstance(obj, dict):
        return {k: _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


@dataclass
class LimitOrder:
    """Ordine limite dell'umano, in attesa di essere colpito.

    Vive nel futuro: a ogni settimana successiva controlliamo se il MINIMO
    della settimana è sceso fino al limite (riempire sulla chiusura sarebbe
    barare: il minimo può essere stato toccato e poi rimbalzato).
    Muore in tre modi: eseguito, scaduto, o invalidato dal setup che esce
    di Fase 2 — quest'ultimo è ciò che lo rende fedele al metodo.
    """
    ticker: str
    limit: float
    shares: float
    stop: float
    placed: pd.Timestamp
    expires: pd.Timestamp


@dataclass
class PendingSignal:
    """Un segnale in attesa di decisione a una certa settimana."""
    ticker: str
    date: pd.Timestamp
    entry: float
    stop: float                        # stop del bot (buffer sulla resistenza)
    base_len: int
    vol_ratio: float
    mansfield: float
    pivot_stop: float | None = None    # ultimo minimo relativo prima del breakout

    def to_public(self, alias: str, blind: bool) -> dict[str, Any]:
        d = {"alias": alias, "entry": round(self.entry, 2), "stop": round(self.stop, 2),
             "base_len": self.base_len, "vol_ratio": round(self.vol_ratio, 2),
             "mansfield": round(self.mansfield, 2),
             "risk_pct": round((self.entry - self.stop) / self.entry * 100, 1)}
        if self.pivot_stop is not None:
            d["pivot_stop"] = round(self.pivot_stop, 2)
            d["pivot_risk_pct"] = round((self.entry - self.pivot_stop) / self.entry * 100, 1)
        else:
            d["pivot_stop"] = None      # nessun minimo relativo: opzione non disponibile
        if not blind:
            d["ticker"] = self.ticker
            d["date"] = self.date.strftime("%Y-%m-%d")
        return d


class Session:
    """Una partita. Vive in memoria lato server (per il preview è sufficiente)."""

    # Stop discrezionale: quanto può stare sotto l'ingresso. Il tetto era 10%, ma
    # è un handicap: gli stop STRUTTURALI del bot (pivot su barre settimanali) stanno
    # a 15-36% sotto l'ingresso (misurati: STOCK-41 16.1%, MDSO 28.5%, BIIB 36.5%).
    # Con un tetto al 10% il giocatore verrebbe stoppato dal rumore normale e
    # perderebbe per il vincolo, non per la selezione. 35% copre l'intervallo vero.
    MANUAL_RANGE = (0.05, 0.35)   # stop discrezionale: tra 5% e 35% sotto l'ingresso

    def __init__(self, cfg: dict, blind: bool = True, seed_alias: int = 0) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.cfg = cfg
        self.blind = blind
        self.finished = False

        store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
        self._prepare(store)

        pf_cfg = cfg["portfolio"]
        mk = lambda: Portfolio(pf_cfg["initial_capital"], pf_cfg["risk_per_trade"],
                               pf_cfg["max_position_pct"], pf_cfg["max_positions"],
                               pf_cfg["commission_bps"], pf_cfg["slippage_bps"])
        self.pf: dict[Actor, Portfolio] = {"human": mk(), "bot": mk()}
        self.equity_hist: dict[Actor, list[tuple]] = {"human": [], "bot": []}

        self._wi = 0                       # indice sulla lista delle settimane
        self._pending: list[PendingSignal] = []
        self.limit_orders: list[LimitOrder] = []   # ordini limite dell'umano
        self.stop_modes: dict[str, str] = {}       # ticker -> modalità stop scelta
        self.week_counter = 0

        # --- registro della partita (per l'export e l'analisi a posteriori) ---
        # Ogni segnale proposto viene registrato col CONTESTO che il giocatore
        # aveva davanti: senza quello, rivedendo la partita non si capisce
        # perché si era deciso così.
        self.journal: list[dict] = []              # una riga per segnale proposto
        self.event_log: list[dict] = []            # tutti gli eventi, con settimana
        self._journal_idx: dict[tuple, int] = {}   # (ticker, date) -> indice journal
        self.started_at = pd.Timestamp.now("UTC").isoformat()

    # ------------------------------------------------------------------
    def _prepare(self, store: DataStore) -> None:
        cfg, bt, st = self.cfg, self.cfg["backtest"], self.cfg["stages"]
        universe = load_universe(cfg)
        market = next(iter(universe))
        spec = universe[market]

        b_daily = store.get_with_warmup(spec["benchmark"], bt["start"], bt["end"])
        self.bench = enrich_weekly(to_weekly(b_daily), None, st["ma_weeks"], st["slope_lookback"])
        self.market_stage = classify_stages(self.bench, st["flat_slope"])

        self.data: dict[str, pd.DataFrame] = {}
        self.signals_by_week: dict[pd.Timestamp, list[PendingSignal]] = {}
        # Settimanale dalla CACHE (stessa chiave del backtest): senza, avviare una
        # partita ricalcolava il settimanale di ogni titolo — 38 secondi su 118
        # titoli, e la UI sembrava bloccata. Con la cache è quasi istantaneo.
        wkey = store.weekly_key(spec["benchmark"], st["ma_weeks"], st["slope_lookback"],
                                st["flat_slope"], bt["start"], bt["end"])

        def _compute_weekly(daily):
            wk = enrich_weekly(to_weekly(daily), self.bench,
                               st["ma_weeks"], st["slope_lookback"])
            return prepare_ticker(wk, st["flat_slope"])

        for tk in spec["tickers"]:
            df = store.get_weekly_enriched(tk, bt["start"], bt["end"], wkey,
                                           _compute_weekly).copy()
            df["swing_low"] = confirmed_pivot_lows(df, cfg["exits"].get("pivot_k", 2))
            self.data[tk] = df
            for s in detect_signals(tk, df, self.market_stage, cfg["signal"],
                                    cfg["exits"], st["flat_slope"]):
                if pd.Timestamp(bt["start"]) <= s.date <= pd.Timestamp(bt["end"]):
                    # ultimo minimo relativo CONFERMATO prima del breakout:
                    # l'ancora dello stop secondo il libro (puo' essere None)
                    pivot = last_confirmed_pivot_before(
                        df, s.date, cfg["exits"].get("pivot_k", 2))
                    if pivot is not None and pivot >= s.entry:
                        pivot = None       # sopra l'ingresso: inutilizzabile
                    ps = PendingSignal(s.ticker, s.date, s.entry, s.stop,
                                       s.base_len, s.vol_ratio, s.mansfield,
                                       pivot_stop=pivot)
                    self.signals_by_week.setdefault(s.date, []).append(ps)

        # calendario comune + alias stabili per la modalità cieca
        self.weeks = sorted(set().union(*[set(df.loc[bt["start"]:bt["end"]].index)
                                          for df in self.data.values()]))
        self.alias = {tk: f"STOCK-{i:02d}" for i, tk in enumerate(sorted(self.data))}
        self.alias_to_ticker = {v: k for k, v in self.alias.items()}

    # ------------------------------------------------------------------
    def _prices(self, wk_date: pd.Timestamp) -> dict[str, float]:
        out = {}
        for tk, df in self.data.items():
            if df.index.asof(wk_date) is not pd.NaT:
                out[tk] = float(df["adj_close"].asof(wk_date))
        return out

    def _apply_exits(self, wk_date: pd.Timestamp) -> list[dict]:
        """Applica trailing + stop + MA breakdown a ENTRAMBI i portafogli.
        Le uscite non sono una decisione discrezionale in questa v1: sono
        regole identiche per umano e bot (il gioco è sull'INGRESSO)."""
        events = []
        cfg = self.cfg
        k = cfg["exits"].get("pivot_k", 2)
        buf = cfg["exits"].get("swing_buffer", 0.03)
        for actor, pf in self.pf.items():
            for tk in list(pf.positions):
                df = self.data[tk]
                if wk_date not in df.index:
                    continue
                row = df.loc[wk_date]
                pos = pf.positions[tk]
                # trailing sui minimi crescenti
                if cfg["exits"].get("trailing_swing", False):
                    sw = row["swing_low"]
                    if not np.isnan(sw):
                        pivot_week = df.index[df.index.get_loc(wk_date) - k]
                        if pivot_week > pos.entry_date:
                            ns = float(sw) * (1 - buf)
                            if ns > pos.stop:
                                pos.stop = ns
                                pos.raised = True
                close = float(row["adj_close"])
                if close <= pos.stop:
                    tr = pf.close(tk, wk_date, close, "trailing_stop" if pos.raised else "stop")
                    events.append({"actor": actor, "kind": "exit", "alias": self.alias[tk],
                                   "reason": tr.reason, "ret": round(tr.ret, 4)})
                elif cfg["exits"]["ma_breakdown"] and not np.isnan(row["ma"]) \
                        and close < row["ma"] and row["ma_slope"] < 0:
                    tr = pf.close(tk, wk_date, close, "ma_breakdown")
                    events.append({"actor": actor, "kind": "exit", "alias": self.alias[tk],
                                   "reason": tr.reason, "ret": round(tr.ret, 4)})
        return events

    def _process_limit_orders(self, wk_date: pd.Timestamp) -> list[dict]:
        """Processa gli ordini limite dell'umano. Tre esiti possibili:
        eseguito (il MINIMO della settimana ha toccato il limite), scaduto,
        o annullato perché il setup è invalidato.

        Due sottigliezze imparate a caro prezzo:
        1) l'ordine NON si valuta nella settimana in cui è stato piazzato
           (altrimenti si autoesegue o si autoannulla sul breakout stesso);
        2) l'invalidazione guarda la STRUTTURA (prezzo che chiude sotto lo
           stop, o sotto una MA30 in discesa), non l'etichetta di fase: la
           fase può oscillare tra 1 e 2 nelle settimane subito dopo il
           breakout, e ucciderebbe ogni ordine appena nato.
        """
        events: list[dict] = []
        pf = self.pf["human"]
        still: list[LimitOrder] = []

        for order in self.limit_orders:
            df = self.data[order.ticker]
            if wk_date not in df.index or wk_date <= order.placed:
                still.append(order)          # settimana di piazzamento: si aspetta
                continue
            row = df.loc[wk_date]
            alias = self.alias[order.ticker]
            close = float(row["adj_close"])
            ma = row["ma"]

            # 1) setup invalidato: chiusura sotto lo stop, o sotto MA30 in discesa
            broke_stop = close < order.stop
            broke_ma = (not np.isnan(ma)) and close < float(ma) and row["ma_slope"] < 0
            if broke_stop or broke_ma:
                events.append({"actor": "human", "kind": "limit_cancel", "alias": alias,
                               "reason": "setup invalidato: " +
                                         ("prezzo sotto lo stop" if broke_stop
                                          else "rottura della MA30")})
                continue

            # 2) eseguito: il minimo della settimana ha raggiunto il limite
            if float(row["low"]) <= order.limit:
                if pf.open(order.ticker, wk_date, order.limit, order.stop, order.shares):
                    events.append({"actor": "human", "kind": "limit_fill", "alias": alias,
                                   "price": round(order.limit, 2),
                                   "shares": round(order.shares, 2)})
                else:
                    events.append({"actor": "human", "kind": "limit_cancel", "alias": alias,
                                   "reason": "cassa insufficiente all'esecuzione"})
                continue

            # 3) scaduto
            if wk_date >= order.expires:
                events.append({"actor": "human", "kind": "limit_expire", "alias": alias,
                               "reason": "scaduto senza essere colpito"})
                continue

            still.append(order)

        self.limit_orders = still
        return events

    def _record_journal(self, wk_date: pd.Timestamp, pending: list[PendingSignal],
                        bot_events: list[dict]) -> None:
        """Registra ogni segnale proposto con il contesto che il giocatore vede.
        La decisione umana verrà scritta dopo, da `decide()`."""
        pf_h = self.pf["human"]
        prices = self._prices(wk_date)
        bot_took = {e["alias"] for e in bot_events if e["kind"] == "enter"}
        for ps in pending:
            alias = self.alias[ps.ticker]
            eq = pf_h.equity(prices)
            row = {
                "week": self.week_counter,
                "date": wk_date.strftime("%Y-%m-%d"),
                "alias": alias,
                "ticker": ps.ticker,          # nel JSON esportato: sempre presente
                "entry": round(ps.entry, 2),
                # contesto decisionale: cosa avevi davanti
                "base_len": ps.base_len,
                "vol_ratio": round(ps.vol_ratio, 2),
                "mansfield": round(ps.mansfield, 2),
                "stop_bot": round(ps.stop, 2),
                "stop_pivot": round(ps.pivot_stop, 2) if ps.pivot_stop else None,
                "equity_at_decision": round(eq, 2),
                "free_cash_at_decision": round(pf_h.cash - self.committed_cash(), 2),
                # decisione del bot (immediata, meccanica)
                "bot_action": "buy" if alias in bot_took else "skip",
                "bot_shares": next((e["shares"] for e in bot_events
                                    if e.get("alias") == alias and e["kind"] == "enter"), 0),
                # decisione umana: riempita da decide()
                "human_action": None, "human_shares": None,
                "human_stop": None, "human_stop_mode": None, "human_limit": None,
            }
            self._journal_idx[(ps.ticker, wk_date)] = len(self.journal)
            self.journal.append(row)

    def _log_events(self, wk_date: pd.Timestamp, events: list[dict]) -> None:
        for e in events:
            self.event_log.append({"week": self.week_counter,
                                   "date": wk_date.strftime("%Y-%m-%d"), **e})

    def _bot_decide(self, wk_date: pd.Timestamp, pending: list[PendingSignal]) -> list[dict]:
        """Il bot prende ogni segnale se c'è spazio, sizing a rischio fisso."""
        out = []
        pf = self.pf["bot"]
        prices = self._prices(wk_date)
        for ps in pending:
            eq = pf.equity(prices)
            shares = pf.size_fixed_risk(eq, ps.entry, ps.stop)
            if pf.open(ps.ticker, wk_date, ps.entry, ps.stop, shares):
                out.append({"actor": "bot", "kind": "enter", "alias": self.alias[ps.ticker],
                            "shares": round(shares, 2)})
        return out

    def _record_equity(self, wk_date: pd.Timestamp) -> None:
        prices = self._prices(wk_date)
        for actor, pf in self.pf.items():
            pf.mark_prices(prices)      # last_price aggiornato: equity() onesta
            self.equity_hist[actor].append((wk_date, pf.equity(prices)))

    # ------------------------------------------------------------------
    def advance(self) -> dict[str, Any]:
        """Avanza fino al prossimo evento (segnale da decidere) o alla fine.
        Ritorna lo stato pubblico che il client deve mostrare."""
        cfg_end = pd.Timestamp(self.cfg["backtest"]["end"])
        carried: list[dict] = []   # eventi delle settimane "silenziose": non si perdono
        while self._wi < len(self.weeks):
            wk_date = self.weeks[self._wi]
            self.week_counter += 1
            # ordini limite dell'umano: eseguiti/scaduti/annullati PRIMA delle
            # uscite, così una posizione appena riempita è già gestita dagli stop
            limit_events = self._process_limit_orders(wk_date)
            # il bot agisce sempre alle uscite e ai suoi ingressi
            auto_events = self._apply_exits(wk_date)
            pending = self.signals_by_week.get(wk_date, [])
            bot_events = self._bot_decide(wk_date, pending) if pending else []
            self._record_equity(wk_date)
            week_events = limit_events + auto_events + bot_events
            self._log_events(wk_date, week_events)

            # fermata anche se un ordine limite si è concluso: è un fatto che
            # cambia il tuo portafoglio, devi vederlo
            notable = [e for e in week_events
                       if e["kind"] in ("limit_fill", "limit_cancel", "limit_expire")]
            if pending:
                self._pending = pending
                self._record_journal(wk_date, pending, bot_events)
                self._wi += 1
                return self._state(wk_date, pending, carried + week_events, paused=True)
            if notable:
                self._pending = []
                self._wi += 1
                return self._state(wk_date, [], carried + week_events, paused=True)

            carried.extend(week_events)
            self._wi += 1

        self.finished = True
        return self._state(self.weeks[-1] if self.weeks else cfg_end, [], carried, paused=False)

    def committed_cash(self) -> float:
        """Cassa già impegnata da ordini limite non ancora eseguiti: non è
        spesa, ma non è nemmeno disponibile per un secondo ordine."""
        return sum(o.limit * o.shares for o in self.limit_orders)

    def _write_decision(self, ps: PendingSignal, wk_date: pd.Timestamp, action: str,
                        shares: float | None = None, stop: float | None = None,
                        stop_mode: str | None = None, limit: float | None = None) -> None:
        """Scrive la decisione umana nella riga di journal del segnale.
        Una decisione già andata a buon fine NON viene sovrascritta da un
        tentativo successivo fallito (es. secondo click sullo stesso segnale)."""
        i = self._journal_idx.get((ps.ticker, wk_date))
        if i is None:
            return
        r = self.journal[i]
        if r["human_action"] in ("buy", "limit", "skip"):
            return                       # decisione già presa: è quella che conta
        r["human_action"] = action
        r["human_shares"] = round(shares, 2) if shares else None
        r["human_stop"] = round(stop, 2) if stop else None
        r["human_stop_mode"] = stop_mode
        r["human_limit"] = round(limit, 2) if limit else None

    def resolve_stop(self, ps: PendingSignal, stop_mode: str,
                     stop_price: float | None) -> tuple[float | None, str | None]:
        """Traduce la scelta dell'umano in un prezzo di stop concreto.

        'bot'    -> lo stop del bot (buffer sotto la resistenza)
        'pivot'  -> l'ultimo minimo relativo prima del breakout (puo' mancare)
        'manual' -> discrezionale: tra 5% e 35% SOTTO L'INGRESSO
        Ritorna (stop, errore): uno dei due è sempre None.
        """
        if stop_mode == "bot":
            return ps.stop, None
        if stop_mode == "pivot":
            if ps.pivot_stop is None:
                return None, "nessun minimo relativo confermato per questo segnale"
            return ps.pivot_stop, None
        if stop_mode == "manual":
            if not stop_price or stop_price <= 0:
                return None, "prezzo di stop mancante"
            dist = (ps.entry - stop_price) / ps.entry
            lo, hi = self.MANUAL_RANGE
            if not (lo - 1e-9 <= dist <= hi + 1e-9):
                return None, (f"lo stop discrezionale deve stare tra "
                              f"{lo*100:.0f}% e {hi*100:.0f}% sotto l'ingresso "
                              f"({ps.entry * (1-hi):.2f} – {ps.entry * (1-lo):.2f})")
            return float(stop_price), None
        return None, f"modalità di stop sconosciuta: {stop_mode}"

    def decide(self, alias: str, action: str, shares: float | None,
               limit_price: float | None = None, stop_mode: str = "bot",
               stop_price: float | None = None) -> dict[str, Any]:
        """L'umano decide su un segnale in sospeso:
          'buy'   -> a mercato, sulla chiusura del breakout (+ slippage)
          'limit' -> ordine limite, valido `limit_weeks` settimane
          'skip'  -> passa
        Lo stop è scelto separatamente (bot | pivot | manual).
        """
        wk_date = self.weeks[self._wi - 1]
        ps = next((p for p in self._pending if self.alias[p.ticker] == alias), None)
        if ps is None:
            return {"error": f"nessun segnale in sospeso per {alias}"}
        pf = self.pf["human"]

        if action == "skip":
            self._write_decision(ps, wk_date, "skip")
            return {"ok": True, "skipped": alias}

        stop, err = self.resolve_stop(ps, stop_mode, stop_price)
        if err:
            return {"ok": False, "reason": err}

        if action == "limit":
            if not limit_price or limit_price <= 0:
                return {"ok": False, "reason": "prezzo limite mancante"}
            if limit_price <= stop:
                return {"ok": False,
                        "reason": f"limite {limit_price} sotto lo stop {stop:.2f}: "
                                  "l'ordine nascerebbe già invalidato"}
            sh = float(shares) if shares else pf.size_fixed_risk(
                pf.equity(self._prices(wk_date)), limit_price, stop)
            if sh <= 0:
                return {"ok": False, "reason": "quantità nulla"}
            need = limit_price * sh
            free = pf.cash - self.committed_cash()
            if need > free:
                return {"ok": False,
                        "reason": f"servono {need:,.0f} ma la cassa libera è {free:,.0f}"}
            weeks = self.cfg.get("simulator", {}).get("limit_weeks", 4)
            idx = min(self._wi - 1 + weeks, len(self.weeks) - 1)
            self.limit_orders.append(LimitOrder(
                ticker=ps.ticker, limit=float(limit_price), shares=sh,
                stop=stop, placed=wk_date, expires=self.weeks[idx]))
            self.stop_modes[ps.ticker] = stop_mode
            self._write_decision(ps, wk_date, "limit", sh, stop, stop_mode, limit_price)
            return {"ok": True, "limit_placed": alias, "limit": round(limit_price, 2),
                    "shares": round(sh, 2), "stop": round(stop, 2),
                    "stop_mode": stop_mode, "expires_in_weeks": weeks}

        if action == "buy":
            sh = float(shares) if shares else pf.size_fixed_risk(
                pf.equity(self._prices(wk_date)), ps.entry, stop)
            if pf.open(ps.ticker, wk_date, ps.entry, stop, sh):
                self.stop_modes[ps.ticker] = stop_mode
                self._write_decision(ps, wk_date, "buy", sh, stop, stop_mode)
                return {"ok": True, "entered": alias, "shares": round(sh, 2),
                        "stop": round(stop, 2), "stop_mode": stop_mode}
            # ingresso rifiutato: lo registro comunque, altrimenti il journal
            # resta con un buco e l'analisi a posteriori non si spiega la riga
            reason = ("posizione già aperta" if ps.ticker in pf.positions
                      else "cassa insufficiente")
            self._write_decision(ps, wk_date, f"rejected:{reason}")
            return {"ok": False, "reason": f"{reason}"}

        return {"ok": False, "reason": f"azione sconosciuta: {action}"}

    # ------------------------------------------------------------------
    def _state(self, wk_date, pending, events, paused) -> dict[str, Any]:
        prices = self._prices(wk_date)
        pf_h = self.pf["human"]
        committed = self.committed_cash()
        pub = {
            "session_id": self.id, "blind": self.blind, "paused": paused,
            "finished": self.finished, "week_counter": self.week_counter,
            "progress": round(self._wi / max(len(self.weeks), 1) * 100, 1),
            "market_stage": int(self.market_stage.asof(wk_date))
                            if self.market_stage.asof(wk_date) == self.market_stage.asof(wk_date) else 0,
            "equity": {a: round(pf.equity(prices), 2) for a, pf in self.pf.items()},
            "cash": {a: round(pf.cash, 2) for a, pf in self.pf.items()},
            "committed_cash": round(committed, 2),
            "free_cash": round(pf_h.cash - committed, 2),
            "invested_pct": round((1 - pf_h.cash / max(pf_h.equity(prices), 1)) * 100, 1),
            "n_positions": {a: len(pf.positions) for a, pf in self.pf.items()},
            "max_positions": self.cfg["portfolio"]["max_positions"],
            "open_positions": {a: [self._pos_public(p, prices) for p in pf.positions.values()]
                               for a, pf in self.pf.items()},
            "limit_orders": [self._limit_public(o, wk_date) for o in self.limit_orders],
            "events": events,
            "pending": [self._pending_public(p, pf_h, prices) for p in pending],
        }
        if not self.blind:
            pub["date"] = wk_date.strftime("%Y-%m-%d")
        if self.finished:
            pub["summary"] = self.summary()
        return pub

    def _pending_public(self, ps: PendingSignal, pf: Portfolio, prices: dict) -> dict:
        """Segnale + le tre opzioni di stop, ciascuna con la quantità che
        manterrebbe il rischio al target (1% dell'equity). Il giocatore resta
        libero di ignorarla, ma la vede."""
        d = ps.to_public(self.alias[ps.ticker], self.blind)
        eq = pf.equity(prices)
        d["bot_shares"] = round(pf.size_fixed_risk(eq, ps.entry, ps.stop), 2)
        d["equity"] = round(eq, 2)
        d["risk_target_pct"] = self.cfg["portfolio"]["risk_per_trade"] * 100

        lo, hi = self.MANUAL_RANGE
        opts = {
            "bot": {"stop": round(ps.stop, 2), "available": True},
            "pivot": {"stop": round(ps.pivot_stop, 2) if ps.pivot_stop else None,
                      "available": ps.pivot_stop is not None},
            "manual": {"min": round(ps.entry * (1 - hi), 2),
                       "max": round(ps.entry * (1 - lo), 2),
                       "default": round(ps.entry * (1 - (lo + hi) / 2), 2),
                       "available": True},
        }
        # per bot e pivot: rischio % e quantità che tiene il rischio all'1%
        for key in ("bot", "pivot"):
            st = opts[key]["stop"]
            if st:
                opts[key]["risk_pct"] = round((ps.entry - st) / ps.entry * 100, 1)
                opts[key]["suggested_shares"] = round(pf.size_fixed_risk(eq, ps.entry, st), 2)
        d["stop_options"] = opts
        return d

    def _limit_public(self, o: LimitOrder, wk_date: pd.Timestamp) -> dict:
        weeks_left = max(0, len([w for w in self.weeks if wk_date < w <= o.expires]))
        return {"alias": self.alias[o.ticker], "limit": round(o.limit, 2),
                "shares": round(o.shares, 2), "stop": round(o.stop, 2),
                "value": round(o.limit * o.shares, 2), "weeks_left": weeks_left}

    def _pos_public(self, pos, prices) -> dict:
        cur = prices.get(pos.ticker, pos.entry_price)
        return {"alias": self.alias[pos.ticker], "entry": round(pos.entry_price, 2),
                "stop": round(pos.stop, 2), "shares": round(pos.shares, 2),
                "unreal_pct": round((cur / pos.entry_price - 1) * 100, 1),
                "raised": pos.raised}

    def chart(self, alias: str) -> dict[str, Any]:
        """Serie da disegnare per un titolo, TAGLIATA alla settimana corrente
        (niente futuro visibile: anti look-ahead anche nel grafico)."""
        tk = self.alias_to_ticker.get(alias)
        if tk is None:
            return {"error": "alias sconosciuto"}
        wk_date = self.weeks[min(self._wi, len(self.weeks) - 1)]
        df = self.data[tk].loc[:wk_date]
        vol_avg = df["volume"].shift(1).rolling(4, min_periods=4).mean()

        # JSON non ammette NaN/Infinity: un solo valore non finito fa fallire la
        # risposta e ABBATTE il server. Capita davvero: vol_ratio = volume/media
        # precedente diventa `inf` se una settimana ha volume zero. `np.isfinite`
        # copre NaN e ±inf in un colpo → si manda `null` e il grafico buca il punto.
        def clean(series, nd: int = 2):
            return [None if not np.isfinite(x) else round(float(x), nd) for x in series]

        # OHLC AGGIUSTATO per le candele. Se enrich_weekly non ha prodotto le
        # colonne adj_* (df minimale), si ripiega sulla chiusura: la candela
        # degenera in un trattino invece di rompere il grafico.
        def ohlc(col: str):
            return clean(df[col] if col in df.columns else df["adj_close"])

        return {
            "alias": alias,
            "t": [d.strftime("%Y-%m-%d") if not self.blind else f"w{i}"
                  for i, d in enumerate(df.index)],
            "price": clean(df["adj_close"]),
            "open": ohlc("adj_open"),
            "high": ohlc("adj_high"),
            "low": ohlc("adj_low"),
            "close": clean(df["adj_close"]),
            "ma": clean(df["ma"]),
            "stage": [int(x) if np.isfinite(x) else 0 for x in df["stage"]],
            "volume": clean(df["volume"], 0),
            "vol_avg": clean(vol_avg, 0),
            "vol_ratio": clean(df["vol_ratio"]),
            "mansfield": clean(df["mansfield"]),
            "resistance": clean(df["resistance"]),
        }

    def divergences(self) -> dict[str, Any]:
        """I segnali dove umano e bot hanno scelto diversamente, e come sono andati.
        È la domanda a cui il simulatore esiste per rispondere: il giudizio
        umano aggiunge o toglie valore rispetto alle regole meccaniche?"""
        # esito per ticker: rendimento del trade del bot su quel segnale
        bot_ret = {t.ticker: t.ret for t in self.pf["bot"].trades}
        human_ret = {t.ticker: t.ret for t in self.pf["human"].trades}

        missed, avoided, agreed, unresolved = [], [], [], []
        for r in self.journal:
            h, b = r["human_action"], r["bot_action"]
            took_h = h in ("buy", "limit")
            took_b = b == "buy"
            row = {"week": r["week"], "alias": r["alias"], "ticker": r["ticker"],
                   "entry": r["entry"], "vol_ratio": r["vol_ratio"],
                   "mansfield": r["mansfield"],
                   "bot_ret_pct": round(bot_ret[r["ticker"]] * 100, 1)
                                  if r["ticker"] in bot_ret else None,
                   "human_ret_pct": round(human_ret[r["ticker"]] * 100, 1)
                                    if r["ticker"] in human_ret else None}
            if h is None or (isinstance(h, str) and h.startswith("rejected")):
                # non decisi o rifiutati (es. titolo già in portafoglio):
                # non sono giudizio umano, non inquinano le divergenze
                row["why"] = h or "non deciso"
                unresolved.append(row)
            elif took_b and not took_h:
                missed.append(row)      # il bot ha preso, tu hai passato
            elif took_h and not took_b:
                avoided.append(row)     # tu hai preso, il bot no
            elif took_h and took_b:
                agreed.append(row)

        def _score(rows, key):
            vals = [r[key] for r in rows if r[key] is not None]
            if not vals:
                return None
            return {"n": len(vals), "avg_ret_pct": round(sum(vals) / len(vals), 1),
                    "winners": sum(1 for v in vals if v > 0)}

        return {
            "missed": {"rows": missed, "score": _score(missed, "bot_ret_pct"),
                       "meaning": "segnali che hai passato e il bot ha preso"},
            "taken_alone": {"rows": avoided, "score": _score(avoided, "human_ret_pct"),
                            "meaning": "segnali che hai preso e il bot no"},
            "agreed": {"n": len(agreed),
                       "meaning": "segnali su cui eravate d'accordo"},
            "unresolved": {"rows": unresolved, "n": len(unresolved),
                           "meaning": "segnali non decisi o rifiutati (non giudizio umano)"},
        }

    def export(self) -> dict[str, Any]:
        """Partita completa in forma serializzabile: metadati, journal delle
        decisioni col contesto, operazioni chiuse, eventi, curve, divergenze."""
        def trades(actor: Actor) -> list[dict]:
            return [{"ticker": t.ticker, "alias": self.alias.get(t.ticker, t.ticker),
                     "entry_date": t.entry_date.strftime("%Y-%m-%d"),
                     "exit_date": t.exit_date.strftime("%Y-%m-%d"),
                     "entry_price": round(t.entry_price, 2),
                     "exit_price": round(t.exit_price, 2),
                     "shares": round(t.shares, 2), "reason": t.reason,
                     "pnl": round(t.pnl, 2), "ret_pct": round(t.ret * 100, 2),
                     "stop_mode": self.stop_modes.get(t.ticker) if actor == "human" else "bot"}
                    for t in self.pf[actor].trades]

        open_pos = {a: [{"ticker": p.ticker, "alias": self.alias[p.ticker],
                         "entry_date": p.entry_date.strftime("%Y-%m-%d"),
                         "entry_price": round(p.entry_price, 2),
                         "shares": round(p.shares, 2), "stop": round(p.stop, 2),
                         "trailing_raised": p.raised}
                        for p in pf.positions.values()]
                    for a, pf in self.pf.items()}

        payload = {
            "meta": {
                "session_id": self.id,
                "started_at": self.started_at,
                "exported_at": pd.Timestamp.now("UTC").isoformat(),
                "blind_mode": self.blind,
                "provider": self.cfg["provider"],
                "period": {"start": self.cfg["backtest"]["start"],
                           "end": self.cfg["backtest"]["end"]},
                "weeks_played": self.week_counter,
                "finished": self.finished,
                "config": {k: self.cfg[k] for k in
                           ("stages", "signal", "exits", "portfolio")},
                "alias_map": self.alias_to_ticker,
                "warning": ("Dati sintetici: i risultati validano il motore, "
                            "non la strategia." if self.cfg["provider"] == "synthetic"
                            else None),
            },
            "summary": self.summary() if self.finished else None,
            "journal": self.journal,
            "trades": {"human": trades("human"), "bot": trades("bot")},
            "open_positions": open_pos,
            "pending_limit_orders": [
                {"alias": self.alias[o.ticker], "ticker": o.ticker,
                 "limit": round(o.limit, 2), "shares": round(o.shares, 2),
                 "stop": round(o.stop, 2), "placed": o.placed.strftime("%Y-%m-%d"),
                 "expires": o.expires.strftime("%Y-%m-%d")}
                for o in self.limit_orders],
            "events": self.event_log,
            "divergences": self.divergences(),
            "equity_curves": self.equity_curves(),
        }
        return _py(payload)      # sanificazione: niente numpy nel JSON

    def export_trades_csv(self) -> str:
        """Operazioni di umano e bot in CSV, apribile in Excel."""
        import csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["actor", "alias", "ticker", "entry_date", "exit_date",
                    "entry_price", "exit_price", "shares", "stop_mode",
                    "reason", "pnl", "ret_pct"])
        for actor in ("human", "bot"):
            for t in self.pf[actor].trades:
                w.writerow([actor, self.alias.get(t.ticker, t.ticker), t.ticker,
                            t.entry_date.date(), t.exit_date.date(),
                            round(t.entry_price, 2), round(t.exit_price, 2),
                            round(t.shares, 2),
                            self.stop_modes.get(t.ticker, "bot") if actor == "human" else "bot",
                            t.reason, round(t.pnl, 2), round(t.ret * 100, 2)])
        return buf.getvalue()

    def summary(self) -> dict[str, Any]:
        prices = self._prices(self.weeks[-1]) if self.weeks else {}
        out = {"stage_names": STAGE_NAMES, "reveal": self.alias_to_ticker if self.blind else None}
        # quali modalità di stop hai usato, e come sono andate: il feedback che
        # trasforma il simulatore in uno strumento di apprendimento
        by_mode: dict[str, dict] = {}
        for t in self.pf["human"].trades:
            m = self.stop_modes.get(t.ticker, "bot")
            b = by_mode.setdefault(m, {"n": 0, "wins": 0, "sum_ret": 0.0})
            b["n"] += 1
            b["wins"] += 1 if t.pnl > 0 else 0
            b["sum_ret"] += t.ret
        for m, b in by_mode.items():
            b["hit_rate_pct"] = round(b["wins"] / b["n"] * 100, 1)
            b["avg_ret_pct"] = round(b["sum_ret"] / b["n"] * 100, 1)
            b.pop("sum_ret")
        out["human_stop_modes"] = by_mode
        out["divergences"] = self.divergences()
        for actor, pf in self.pf.items():
            eq0 = self.cfg["portfolio"]["initial_capital"]
            eqN = pf.equity(prices)
            wins = [t for t in pf.trades if t.pnl > 0]
            curve = pd.Series(dict(self.equity_hist[actor])).sort_index()
            dd = (curve / curve.cummax() - 1).min() if len(curve) > 1 else 0.0
            out[actor] = {
                "final_equity": round(eqN, 2),
                "total_return_pct": round((eqN / eq0 - 1) * 100, 1),
                "n_trades": len(pf.trades),
                "hit_rate_pct": round(len(wins) / len(pf.trades) * 100, 1) if pf.trades else None,
                "max_dd_pct": round(dd * 100, 1),
            }
        return out

    def equity_curves(self) -> dict[str, Any]:
        """Curve equity in tipi Python NATIVI (vedi _py): numpy non è
        serializzabile in JSON."""
        out = {}
        for actor in ("human", "bot"):
            c = pd.Series(dict(self.equity_hist[actor])).sort_index()
            out[actor] = {"t": [d.strftime("%Y-%m-%d") for d in c.index],
                          "equity": [round(float(x), 2) for x in c.values]}
        b = self.bench["adj_close"]
        b = b.loc[self.cfg["backtest"]["start"]:self.cfg["backtest"]["end"]]
        b = b / b.iloc[0] * self.cfg["portfolio"]["initial_capital"]
        out["benchmark"] = {"t": [d.strftime("%Y-%m-%d") for d in b.index],
                            "equity": [round(float(x), 2) for x in b.values]}
        return out
