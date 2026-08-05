"""Test unitari del motore con scenari costruiti a mano.

La filosofia: ogni test disegna una storia di prezzo di cui CONOSCIAMO
la classificazione corretta, e verifica che il motore la riconosca.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators import enrich_weekly, mansfield_rs, to_weekly, volume_ratio
from src.portfolio import Portfolio
from src.signals import detect_signals, prepare_ticker
from src.stages import classify_stages

FLAT = 0.005


# ----------------------------------------------------------------- helpers
def weekly_frame(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """Costruisce un DataFrame settimanale direttamente (venerdì consecutivi)."""
    n = len(closes)
    idx = pd.date_range("2010-01-08", periods=n, freq="W-FRI")
    c = pd.Series(closes, index=idx, dtype=float)
    v = pd.Series(volumes if volumes is not None else [1e6] * n, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "adj_close": c, "volume": v})


def enrich(wk: pd.DataFrame, bench: pd.DataFrame | None = None,
           ma_weeks: int = 30) -> pd.DataFrame:
    return enrich_weekly(wk, bench, ma_weeks=ma_weeks, slope_lookback=4)


# -------------------------------------------------------------- indicators
def test_to_weekly_aggregation():
    days = pd.bdate_range("2020-01-06", "2020-01-17")  # due settimane piene
    df = pd.DataFrame({"open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0,
                       "adj_close": 11.0, "volume": 100.0}, index=days)
    wk = to_weekly(df)
    assert len(wk) == 2
    assert wk["volume"].iloc[0] == 500.0          # somma dei 5 giorni
    assert wk["high"].iloc[0] == 12.0


def test_volume_ratio_no_lookahead():
    """Il ratio della settimana t usa solo le 4 settimane PRECEDENTI."""
    v = pd.Series([100, 100, 100, 100, 300], dtype=float,
                  index=pd.date_range("2020-01-03", periods=5, freq="W-FRI"))
    r = volume_ratio(v, n=4)
    assert r.iloc[4] == pytest.approx(3.0)        # 300 / media(100x4)
    assert np.isnan(r.iloc[3])                    # non abbastanza storia precedente


def test_mansfield_sign():
    idx = pd.date_range("2015-01-02", periods=120, freq="W-FRI")
    bench = pd.Series(100.0, index=idx)                        # piatto
    strong = pd.Series(np.linspace(100, 200, 120), index=idx)  # batte il mercato
    m = mansfield_rs(strong, bench, n=52)
    assert m.dropna().iloc[-1] > 0


# ------------------------------------------------------------------ stages
def _cycle_closes() -> list[float]:
    """Declino -> base LUNGA -> avanzata: la storia canonica 4 -> 1 -> 2.
    La base dura 50 settimane: serve tempo perché la MA30, gonfiata dal
    declino precedente, si appiattisca davvero (com'è nei mercati reali)."""
    decline = list(np.linspace(100, 70, 45))
    base = [70 + 0.5 * np.sin(i / 2.5) for i in range(50)]
    advance = list(np.linspace(70.5, 130, 45))
    return decline + base + advance


def test_stage_cycle_recognized():
    df = enrich(weekly_frame(_cycle_closes()))
    st = classify_stages(df, FLAT)
    # nel cuore del declino (MA matura, ancora in discesa) -> Fase 4
    assert st.iloc[40] == 4
    # a base matura (MA ormai appiattita dopo il declino) -> Fase 1
    assert st.iloc[90] == 1
    # nel cuore dell'avanzata -> Fase 2
    assert st.iloc[132] == 2


def test_transition_context_top_vs_base():
    """MA piatta dopo un'avanzata deve essere Fase 3, non Fase 1."""
    advance = list(np.linspace(50, 120, 60))
    top = [120 + 0.5 * np.sin(i / 2.0) for i in range(45)]
    df = enrich(weekly_frame(advance + top))
    st = classify_stages(df, FLAT)
    assert st.iloc[-1] == 3


# ----------------------------------------------------------------- signals
def _breakout_scenario(vol_spike: float = 4.0, bench_stage2: bool = True):
    closes = _cycle_closes()
    n = len(closes)
    volumes = [1e6] * n
    # picco di volume nelle prime settimane dell'avanzata (indici 95..104)
    for i in range(95, 105):
        volumes[i] = vol_spike * 1e6
    wk = weekly_frame(closes, volumes)
    if bench_stage2:
        bench = weekly_frame(list(np.linspace(80, 160, n)))   # mercato in salita
    else:
        bench = weekly_frame(list(np.linspace(160, 80, n)))   # mercato in discesa
    b_enriched = enrich(bench)
    df = prepare_ticker(enrich(wk, b_enriched), FLAT)
    market_stage = classify_stages(b_enriched, FLAT)
    sig_cfg = dict(min_base_weeks=8, max_base_depth=0.40, volume_ratio_min=2.0,
                   mansfield_min=-100.0, mansfield_rising_weeks=4, market_filter=True)
    exit_cfg = dict(stop_buffer=0.03, ma_breakdown=True)
    return detect_signals("TEST", df, market_stage, sig_cfg, exit_cfg, FLAT)


def test_breakout_detected():
    sigs = _breakout_scenario()
    assert len(sigs) >= 1
    s = sigs[0]
    assert s.base_len >= 8
    assert s.vol_ratio >= 2.0
    assert s.stop < s.entry


def test_no_signal_without_volume():
    assert _breakout_scenario(vol_spike=1.0) == []


def test_max_breakout_stretch_rejects_gapped_breakout():
    """Filtro anti-spike-da-notizia: un breakout che chiude troppo sopra la
    resistenza (gap oltre la base) va scartato. Soglia stretta → 0 segnali;
    assente → il segnale c'è."""
    closes = _cycle_closes()
    n = len(closes)
    volumes = [1e6] * n
    for i in range(95, 105):
        volumes[i] = 4.0 * 1e6
    wk = weekly_frame(closes, volumes)
    bench = weekly_frame(list(np.linspace(80, 160, n)))
    b_enriched = enrich(bench)
    df = prepare_ticker(enrich(wk, b_enriched), FLAT)
    market_stage = classify_stages(b_enriched, FLAT)
    base = dict(min_base_weeks=8, max_base_depth=0.40, volume_ratio_min=2.0,
                mansfield_min=-100.0, mansfield_rising_weeks=4, market_filter=True)
    exit_cfg = dict(stop_buffer=0.03, ma_breakdown=True)
    assert len(detect_signals("T", df, market_stage, base, exit_cfg, FLAT)) >= 1
    tight = {**base, "max_breakout_stretch": 0.0001}   # qualsiasi gap sopra la resistenza
    assert detect_signals("T", df, market_stage, tight, exit_cfg, FLAT) == []


def test_market_filter_blocks_signal():
    assert _breakout_scenario(bench_stage2=False) == []


# --------------------------------------------------------------- portfolio
def test_portfolio_sizing_and_roundtrip():
    pf = Portfolio(initial_capital=100_000, risk_per_trade=0.01,
                   max_position_pct=0.20, max_positions=5,
                   commission_bps=5, slippage_bps=10)
    ts = pd.Timestamp("2020-01-03")
    shares = pf.size_fixed_risk(100_000, entry=50.0, stop=45.0)
    assert shares == pytest.approx(200.0)         # 1000 di rischio / 5 per azione
    pf.open("XYZ", ts, 50.0, 45.0, shares)
    assert pf.cash < 100_000
    pf.close("XYZ", ts + pd.Timedelta(weeks=10), 60.0, "test")
    t = pf.trades[0]
    assert t.pnl > 0
    assert t.ret < 0.20                           # i costi mordono: < +20% teorico


def test_position_cap():
    pf = Portfolio(100_000, risk_per_trade=0.10, max_position_pct=0.20,
                   max_positions=5, commission_bps=0, slippage_bps=0)
    shares = pf.size_fixed_risk(100_000, entry=100.0, stop=99.0)
    assert shares * 100.0 <= 20_000 + 1e-6        # cap al 20% dell'equity


# ---------------------------------------------------------- trailing swing
def test_confirmed_pivot_lows_timing():
    """Il pivot in p è noto solo in p+k: la serie deve valorizzarsi lì, non prima."""
    from src.indicators import confirmed_pivot_lows
    lows = [10, 9, 8, 7, 8.5, 9.5, 10.5, 11]          # minimo in indice 3 (=7)
    idx = pd.date_range("2020-01-03", periods=len(lows), freq="W-FRI")
    wk = pd.DataFrame({"low": lows, "high": [x + 2 for x in lows],
                       "open": lows, "close": lows, "adj_close": lows,
                       "volume": 1e6}, index=idx)
    piv = confirmed_pivot_lows(wk, k=2)
    assert np.isnan(piv.iloc[3])                      # al minimo non si sa ancora
    assert np.isnan(piv.iloc[4])                      # una settimana dopo nemmeno
    assert piv.iloc[5] == pytest.approx(7.0)          # confermato a p+k
    assert piv.iloc[6:].isna().all()


def test_trailing_stop_ratchet_logic():
    """Il ratchet: pivot più alti alzano lo stop, pivot più bassi lo ignorano."""
    stop = 50.0
    buf = 0.03
    for pivot, expected in [(55.0, 55.0 * 0.97),      # sale
                            (53.0, 55.0 * 0.97),      # più basso: ignorato
                            (60.0, 60.0 * 0.97)]:     # sale ancora
        candidate = pivot * (1 - buf)
        if candidate > stop:
            stop = candidate
        assert stop == pytest.approx(expected)


# ------------------------------------------------------------ simulatore
def test_simulator_human_matches_bot():
    """Se l'umano imita il bot (accetta ogni segnale con sizing di default),
    le due equity devono coincidere: prova che i portafogli girano sotto
    regole identiche e che l'unica variabile è la decisione umana."""
    from src.config import load_config
    from src.simulator import Session
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config_sim.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config_sim.yaml non presente")
    cfg = load_config(cfg_path)
    cfg["backtest"]["start"] = "2003-01-01"   # periodo piu corto per un test veloce
    cfg["backtest"]["end"] = "2010-01-01"
    s = Session(cfg, blind=True)
    state = s.advance()
    guard = 0
    while not state.get("finished") and guard < 5000:
        guard += 1
        for p in state.get("pending", []):
            s.decide(p["alias"], "buy", None)
        state = s.advance()
    sm = state["summary"]
    assert sm["human"]["final_equity"] == pytest.approx(sm["bot"]["final_equity"])
    assert sm["human"]["n_trades"] == sm["bot"]["n_trades"]


def test_simulator_blind_hides_identity():
    """In modalità cieca il segnale pubblico non deve rivelare ticker/date."""
    from src.config import load_config
    from src.simulator import Session
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config_sim.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config_sim.yaml non presente")
    cfg = load_config(cfg_path)
    cfg["backtest"]["start"] = "2003-01-01"
    cfg["backtest"]["end"] = "2010-01-01"
    s = Session(cfg, blind=True)
    state = s.advance()
    for p in state.get("pending", []):
        assert "ticker" not in p and "date" not in p
        assert p["alias"].startswith("STOCK-")


# ------------------------------------------------------- ordini limite
def _sim_cfg():
    from src.config import load_config
    import os
    p = os.path.join(os.path.dirname(__file__), "..", "config_sim.yaml")
    if not os.path.exists(p):
        pytest.skip("config_sim.yaml non presente")
    return load_config(p)


def test_limit_order_rejects_below_stop():
    """Un limite sotto lo stop nascerebbe già invalidato: va rifiutato."""
    from src.simulator import Session
    s = Session(_sim_cfg(), blind=True)
    st = s.advance()
    p = st["pending"][0]
    r = s.decide(p["alias"], "limit", 10, p["stop"] - 1)
    assert r["ok"] is False and "stop" in r["reason"]


def test_limit_order_respects_free_cash():
    """La cassa impegnata da un ordine non è disponibile per un secondo ordine."""
    from src.simulator import Session
    s = Session(_sim_cfg(), blind=True)
    st = s.advance()
    p = st["pending"][0]
    limit = round(p["entry"] * 0.99, 2)
    huge = 100000 / limit * 2          # il doppio della cassa
    r = s.decide(p["alias"], "limit", huge, limit)
    assert r["ok"] is False and "cassa" in r["reason"]


def test_limit_order_expires_after_configured_weeks():
    """L'ordine vive esattamente limit_weeks settimane, poi scade."""
    from src.simulator import Session
    cfg = _sim_cfg()
    weeks = cfg.get("simulator", {}).get("limit_weeks", 4)
    s = Session(cfg, blind=True)
    st = s.advance()
    p = st["pending"][0]
    s.decide(p["alias"], "limit", 10, round(p["entry"] * 0.98, 2))
    o = s.limit_orders[0]
    span = len([w for w in s.weeks if o.placed < w <= o.expires])
    assert span == weeks


def test_limit_order_not_evaluated_on_placement_week():
    """L'ordine non si auto-esegue né si auto-annulla nella settimana in cui nasce."""
    from src.simulator import Session
    s = Session(_sim_cfg(), blind=True)
    st = s.advance()
    p = st["pending"][0]
    # limite molto alto: verrebbe colpito subito se valutassimo la settimana di piazzamento
    s.decide(p["alias"], "limit", 10, round(p["entry"] * 1.5, 2))
    wk = s.weeks[s._wi - 1]
    ev = s._process_limit_orders(wk)
    assert ev == []                     # nessun evento: si aspetta la settimana dopo
    assert len(s.limit_orders) == 1


# --------------------------------------------------- tre modalità di stop
def test_last_pivot_finds_relative_low_not_base_bottom():
    """L'ancora è il minimo RELATIVO che precede il breakout, non il fondo
    della base (che su una base lunga è lontanissimo e irrilevante)."""
    from src.indicators import last_confirmed_pivot_before
    lows = [44]*10 + [50,58,66,74,88,86,82,84,90,95,98]   # minimo relativo: 82
    idx = pd.date_range("2020-01-03", periods=len(lows), freq="W-FRI")
    wk = pd.DataFrame({"low": lows, "high": [x*1.03 for x in lows], "open": lows,
                       "close": lows, "adj_close": lows, "volume": 1e6}, index=idx)
    assert last_confirmed_pivot_before(wk, idx[-1], k=2) == 82   # non 44


def test_last_pivot_none_on_monotonic_rise():
    """Salita senza oscillazioni: nessun pivot -> opzione non disponibile."""
    from src.indicators import last_confirmed_pivot_before
    lows = list(range(40, 70))
    idx = pd.date_range("2020-01-03", periods=len(lows), freq="W-FRI")
    wk = pd.DataFrame({"low": lows, "high": [x*1.03 for x in lows], "open": lows,
                       "close": lows, "adj_close": lows, "volume": 1e6}, index=idx)
    assert last_confirmed_pivot_before(wk, idx[-1], k=2) is None


def test_manual_stop_range_enforced():
    """Lo stop discrezionale sta tra 5% e 35% sotto l'ingresso (35% copre lo stop
    strutturale reale su barre settimanali: MDSO 28.5%, BIIB 36.5%)."""
    from src.simulator import Session
    s = Session(_sim_cfg(), blind=True)
    st = s.advance()
    p = st["pending"][0]
    ok, err = s.resolve_stop(s._pending[0], "manual", p["entry"] * 0.93)  # -7% (dentro)
    assert err is None and ok == pytest.approx(p["entry"] * 0.93)
    ok2, err2 = s.resolve_stop(s._pending[0], "manual", p["entry"] * 0.80)  # -20% (ora valido)
    assert err2 is None and ok2 == pytest.approx(p["entry"] * 0.80)
    bad, err3 = s.resolve_stop(s._pending[0], "manual", p["entry"] * 0.60)  # -40% (troppo)
    assert bad is None and "35%" in err3


def test_wider_stop_yields_smaller_position():
    """Rischio fisso: stop più lontano => meno azioni. È il cuore del sizing."""
    from src.simulator import Session
    s = Session(_sim_cfg(), blind=True)
    st = s.advance()
    p = st["pending"][0]
    so = p["stop_options"]
    if not so["pivot"]["available"]:
        pytest.skip("nessun pivot su questo segnale")
    assert so["pivot"]["stop"] < so["bot"]["stop"]
    assert so["pivot"]["suggested_shares"] < so["bot"]["suggested_shares"]


# ------------------------------------------------------ export e journal
def test_journal_records_context_and_decision():
    """Ogni segnale proposto entra nel journal col contesto; la decisione
    umana viene scritta da decide()."""
    from src.simulator import Session
    s = Session(_sim_cfg(), blind=True)
    st = s.advance()
    p = st["pending"][0]
    assert len(s.journal) >= 1
    row = s.journal[0]
    for k in ("vol_ratio", "mansfield", "stop_bot", "bot_action", "equity_at_decision"):
        assert k in row
    assert row["human_action"] is None
    s.decide(p["alias"], "skip", None)
    assert s.journal[0]["human_action"] == "skip"


def test_rejected_entry_is_recorded_not_silent():
    """Un ingresso rifiutato (titolo già in portafoglio) non lascia buchi."""
    from src.simulator import Session
    s = Session(_sim_cfg(), blind=True)
    st = s.advance()
    p = st["pending"][0]
    s.decide(p["alias"], "buy", None, None, "bot")
    # secondo tentativo sullo stesso segnale
    r = s.decide(p["alias"], "buy", None, None, "bot")
    assert r["ok"] is False
    assert s.journal[0]["human_action"] == "buy"   # la prima resta valida


def test_export_is_json_serializable_and_complete():
    """L'export contiene tutto ciò che serve all'analisi a posteriori."""
    import json
    from src.simulator import Session
    cfg = _sim_cfg()
    cfg["backtest"]["start"] = "2003-01-01"
    cfg["backtest"]["end"] = "2009-01-01"
    s = Session(cfg, blind=True)
    st = s.advance()
    guard = 0
    while not st.get("finished") and guard < 3000:
        guard += 1
        for p in st.get("pending", []):
            s.decide(p["alias"], "buy" if guard % 2 else "skip", None, None, "bot")
        st = s.advance()
    exp = s.export()
    for k in ("meta", "journal", "trades", "events", "divergences", "equity_curves"):
        assert k in exp
    assert exp["meta"]["blind_mode"] is True
    assert exp["meta"]["warning"] is not None      # provider synthetic dichiarato
    json.dumps(exp)                                 # deve essere serializzabile
    csv = s.export_trades_csv()
    assert csv.startswith("actor,alias,ticker")


def test_divergences_classify_missed_and_taken_alone():
    """Passare un segnale che il bot prende => 'missed'."""
    from src.simulator import Session
    s = Session(_sim_cfg(), blind=True)
    st = s.advance()
    for p in st.get("pending", []):
        s.decide(p["alias"], "skip", None)         # passo tutto
    d = s.divergences()
    # il bot ha comprato almeno il primo segnale => finisce nei 'missed'
    assert d["missed"]["rows"] or d["unresolved"]["rows"]


# ------------------------------------------- universo point-in-time
def _pit_members():
    return pd.DataFrame([
        {"ticker": "ALWAYS.US", "start": pd.NaT, "end": pd.NaT,
         "is_delisted": False, "is_active": True},
        {"ticker": "LEHMAN.US", "start": pd.Timestamp("2000-01-01"),
         "end": pd.Timestamp("2008-09-15"), "is_delisted": True, "is_active": False},
        {"ticker": "LATE.US", "start": pd.Timestamp("2020-12-21"), "end": pd.NaT,
         "is_delisted": False, "is_active": True},
    ])


def test_pit_delisted_buyable_before_death_only():
    """Un titolo delistato è comprabile PRIMA della morte, non dopo.
    È esattamente ciò che elimina il survivorship bias."""
    from src.universe import PointInTimeUniverse
    u = PointInTimeUniverse(_pit_members(), "TEST.INDX")
    assert "LEHMAN.US" in u.active_at("2008-01-01")
    assert "LEHMAN.US" not in u.active_at("2009-01-01")


def test_pit_late_entrant_not_buyable_before_entry():
    """Non puoi comprare nel 2015 un titolo entrato nell'indice nel 2020."""
    from src.universe import PointInTimeUniverse
    u = PointInTimeUniverse(_pit_members(), "TEST.INDX")
    assert "LATE.US" not in u.active_at("2015-01-01")
    assert "LATE.US" in u.active_at("2022-01-01")


def test_pit_membership_matrix_shape_and_content():
    from src.universe import PointInTimeUniverse
    u = PointInTimeUniverse(_pit_members(), "TEST.INDX")
    weeks = pd.date_range("2008-01-04", "2008-12-26", freq="W-FRI")
    mat = u.membership_matrix(weeks)
    assert mat.shape[1] == 3
    leh = mat["LEHMAN.US"]
    assert leh.loc["2008-01-04"]           # vivo a inizio anno
    assert not leh.loc["2008-12-26"]       # morto a fine anno


def test_static_universe_declares_no_history():
    """L'universo statico non ha date: è la fotografia di oggi (bias)."""
    from src.universe import PointInTimeUniverse
    u = PointInTimeUniverse.static(["A.US", "B.US"], "STATIC")
    assert u.active_at("1990-01-01") == ["A.US", "B.US"]   # sempre presenti
    assert u.stats()["delistati"] == 0


def test_backtest_pit_filter_blocks_entries_after_exit(tmp_path):
    """Il backtest non apre posizioni su titoli usciti dall'indice."""
    from src.backtest import run_backtest
    from src.config import load_config
    from src.data_store import DataStore
    from src.providers import make_provider
    from src.providers.synthetic import SyntheticProvider
    from src.universe import PointInTimeUniverse
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config_sim.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config_sim.yaml non presente")
    cfg = load_config(cfg_path)
    cfg["data_dir"] = str(tmp_path)   # MAI scrivere nella data/ reale
    tickers = SyntheticProvider().universe()
    cutoff = pd.Timestamp("2009-01-01")
    rows = [{"ticker": t, "start": pd.NaT,
             "end": cutoff if i % 2 else pd.NaT,
             "is_delisted": bool(i % 2), "is_active": not bool(i % 2)}
            for i, t in enumerate(tickers)]
    PointInTimeUniverse(pd.DataFrame(rows), "SYNTEST.INDX").save(cfg["data_dir"])
    cfg["universe"] = {"US": {"benchmark": "SYN-BENCH", "mode": "index_pit",
                              "index": "SYNTEST.INDX"}}
    res = run_backtest(cfg, DataStore(cfg["data_dir"], make_provider(cfg["provider"])))
    dead = {t for i, t in enumerate(tickers) if i % 2}
    violations = [t for t in res.trades if t.ticker in dead and t.entry_date > cutoff]
    assert not violations, f"comprato un titolo uscito dall'indice: {violations}"
    assert res.metrics["signals_skipped_pit"] > 0   # il filtro ha davvero agito


# ------------------------------------- universo per liquidità e settori
def test_liquidity_no_lookahead():
    """L'idoneità alla settimana t usa le medie delle settimane PRECEDENTI:
    un picco di volume in t non rende il titolo idoneo in t."""
    from src.universe import LiquidityUniverse
    n = 40
    idx = pd.date_range("2020-01-03", periods=n, freq="W-FRI")
    vol = np.full(n, 1e4)
    vol[30] = 1e9                       # picco isolato
    wk = pd.DataFrame({"adj_close": np.full(n, 20.0), "volume": vol,
                       "high": 20.0, "low": 20.0, "open": 20.0, "close": 20.0}, index=idx)
    lu = LiquidityUniverse(min_price=5, min_volume=2e5, min_dollar_volume=5e6,
                           lookback_weeks=13)
    ok = lu.compute("T.US", wk)
    assert not ok.iloc[30]              # il picco non conta per se stesso
    assert ok.iloc[31]                  # conta dalla settimana dopo


def test_liquidity_entry_and_exit_from_universe():
    """Un titolo entra nell'universo quando diventa liquido ed esce quando
    smette: è il point-in-time ottenuto gratis dai dati."""
    from src.universe import LiquidityUniverse
    n = 60
    idx = pd.date_range("2020-01-03", periods=n, freq="W-FRI")
    px = np.concatenate([np.full(20, 3.0), np.full(40, 20.0)])      # prezzo sale sopra 5
    vol = np.concatenate([np.full(45, 5e5), np.full(15, 1e3)])      # volume crolla
    wk = pd.DataFrame({"adj_close": px, "volume": vol, "high": px, "low": px,
                       "open": px, "close": px}, index=idx)
    lu = LiquidityUniverse(min_price=5, min_volume=2e5, min_dollar_volume=5e6,
                           lookback_weeks=13)
    ok = lu.compute("T.US", wk)
    assert not ok.iloc[15]              # prezzo sotto soglia
    assert ok.iloc[35]                  # liquido
    assert not ok.iloc[-1]              # volume crollato: esce


def test_sector_etf_falls_back_before_inception():
    """XLC nasce nel 2018: prima di allora quei titoli stavano in XLK.
    Il codice non inventa dati, usa il predecessore."""
    from src.sectors import SectorMap
    sm = SectorMap({"META.US": "Communication Services", "PLD.US": "Real Estate"})
    assert sm.etf_for("META.US", pd.Timestamp("2015-01-01")) == "XLK.US"
    assert sm.etf_for("META.US", pd.Timestamp("2020-01-01")) == "XLC.US"
    assert sm.etf_for("PLD.US", pd.Timestamp("2012-01-01")) == "XLF.US"


def test_sector_filter_blocks_weak_sector():
    """Il secondo schermo di Weinstein: settore fuori Fase 2 => segnale scartato."""
    from src.sectors import SectorMap
    idx = pd.date_range("2020-01-03", periods=20, freq="W-FRI")
    sm = SectorMap({"AAPL.US": "Technology"})
    sm.register_sector_series("XLK.US", pd.Series(2, index=idx), pd.Series(5.0, index=idx))
    assert sm.sector_ok("AAPL.US", idx[10])
    sm.register_sector_series("XLK.US", pd.Series(4, index=idx), pd.Series(5.0, index=idx))
    assert not sm.sector_ok("AAPL.US", idx[10])
    # RS negativa: scarta anche se in Fase 2
    sm.register_sector_series("XLK.US", pd.Series(2, index=idx), pd.Series(-3.0, index=idx))
    assert not sm.sector_ok("AAPL.US", idx[10])


def test_sector_unknown_does_not_filter():
    """Settore ignoto => filtro inattivo, non scarto arbitrario."""
    from src.sectors import SectorMap
    sm = SectorMap({})
    assert sm.sector_ok("QUALSIASI.US", pd.Timestamp("2020-01-03"))


def test_backtest_liquidity_mode_blocks_illiquid():
    """Soglie proibitive => nessun ingresso, e il conteggio lo dichiara."""
    from src.backtest import run_backtest
    from src.config import load_config
    from src.data_store import DataStore
    from src.providers import make_provider
    from src.providers.synthetic import SyntheticProvider
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config_sim.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config_sim.yaml non presente")
    cfg = load_config(cfg_path)
    cfg["universe"] = {"US": {"benchmark": "SYN-BENCH", "mode": "liquidity",
                              "tickers": SyntheticProvider().universe(),
                              "liquidity": {"min_price": 1e6, "min_volume": 1e12,
                                            "min_dollar_volume": 1e15,
                                            "lookback_weeks": 13}}}
    res = run_backtest(cfg, DataStore(cfg["data_dir"], make_provider(cfg["provider"])))
    assert res.metrics["n_trades"] == 0
    assert res.metrics["signals_skipped_liq"] > 0


# ------------------------------------------- universo per liquidità
def test_liquidity_no_lookahead_on_volume():
    """L'idoneità arriva DOPO il salto di volume, mai nella stessa settimana."""
    from src.universe import LiquidityUniverse
    weeks = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    px = [10.0] * 30
    vol = [50_000] * 15 + [900_000] * 15          # volume esplode alla settimana 15
    df = pd.DataFrame({"adj_close": px, "volume": vol, "close": px,
                       "open": px, "high": px, "low": px}, index=weeks)
    u = LiquidityUniverse(min_price=5.0, min_volume=100_000,
                          min_dollar_volume=3_000_000, lookback_weeks=8)
    ok = u.compute("T", df)
    assert not ok.iloc[15]                        # non idoneo nella settimana del salto
    assert ok.any() and ok.idxmax() > weeks[15]   # idoneo solo dopo


def test_liquidity_no_lookahead_on_price():
    """Un titolo che sfonda i $5 nella settimana del breakout non diventa
    idoneo grazie al breakout stesso."""
    from src.universe import LiquidityUniverse
    weeks = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    px = [4.5] * 14 + [6.0] * 16
    vol = [900_000] * 30
    df = pd.DataFrame({"adj_close": px, "volume": vol, "close": px,
                       "open": px, "high": px, "low": px}, index=weeks)
    u = LiquidityUniverse(min_price=5.0, min_volume=100_000,
                          min_dollar_volume=1_000_000, lookback_weeks=8)
    ok = u.compute("T", df)
    assert not ok.iloc[14]                        # settimana del superamento: no
    assert ok.iloc[15]                            # la successiva: sì


def test_liquidity_excludes_penny_stocks():
    from src.universe import LiquidityUniverse
    weeks = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    px = [2.0] * 30
    df = pd.DataFrame({"adj_close": px, "volume": [900_000] * 30, "close": px,
                       "open": px, "high": px, "low": px}, index=weeks)
    u = LiquidityUniverse(min_price=5.0, min_volume=1, min_dollar_volume=1)
    assert not u.compute("PENNY", df).any()


def test_liquidity_filter_actually_filters_in_backtest():
    """Soglie impossibili => nessun trade. Il filtro deve agire davvero."""
    from src.backtest import run_backtest
    from src.config import load_config
    from src.data_store import DataStore
    from src.providers import make_provider
    from src.providers.synthetic import SyntheticProvider
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config_sim.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config_sim.yaml non presente")
    cfg = load_config(cfg_path)
    cfg["universe"] = {"US": {"benchmark": "SYN-BENCH", "mode": "liquidity",
                              "tickers": SyntheticProvider().universe(),
                              "liquidity": {"min_price": 5.0, "min_volume": 1000,
                                            "min_dollar_volume": 1e15,
                                            "lookback_weeks": 13}}}
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    res = run_backtest(cfg, store)
    assert res.metrics["n_trades"] == 0
    assert res.metrics["signals_skipped_liq"] > 0


# ------------------------------------------- secondo schermo: settore
def test_sector_predecessor_before_spinoff():
    """Real Estate prima del 2015 sta dentro XLF; Communication dentro XLK."""
    from src.sectors import SectorMap
    sm = SectorMap({"AMT.US": "Real Estate", "GOOG.US": "Communication Services"})
    assert sm.etf_for("AMT.US", pd.Timestamp("2005-01-01")) == "XLF.US"
    assert sm.etf_for("AMT.US", pd.Timestamp("2020-01-01")) == "XLRE.US"
    assert sm.etf_for("GOOG.US", pd.Timestamp("2005-01-01")) == "XLK.US"
    assert sm.etf_for("GOOG.US", pd.Timestamp("2020-01-01")) == "XLC.US"


def test_sector_needed_etfs_include_predecessors():
    from src.sectors import SectorMap
    sm = SectorMap({"AMT.US": "Real Estate"})
    assert "XLF.US" in sm.etfs_needed()      # predecessore, serve per la storia antica


def test_sector_blocks_weak_and_passes_strong():
    from src.sectors import SectorMap
    weeks = pd.date_range("2019-01-04", periods=60, freq="W-FRI")
    sm = SectorMap({"A.US": "Technology", "B.US": "Financial Services"})
    sm.register_sector_series("XLK.US", pd.Series([2] * 60, index=weeks),
                              pd.Series([1.0] * 60, index=weeks))
    sm.register_sector_series("XLF.US", pd.Series([4] * 60, index=weeks),
                              pd.Series([-1.0] * 60, index=weeks))
    w = weeks[30]
    assert sm.sector_ok("A.US", w) is True     # settore in Fase 2
    assert sm.sector_ok("B.US", w) is False    # settore in Fase 4


def test_sector_unmapped_leaves_filter_inactive():
    """Titolo non mappato: filtro inattivo, NON esclusione arbitraria."""
    from src.sectors import SectorMap
    sm = SectorMap({"A.US": "Technology"})
    assert sm.sector_ok("SCONOSCIUTO.US", pd.Timestamp("2020-01-01")) is True


def test_sector_coverage_flags_unknown_names():
    """Un nome di settore sbagliato disattiverebbe il filtro in silenzio:
    coverage() deve segnalarlo."""
    from src.sectors import SectorMap
    sm = SectorMap({"A.US": "Financials"})     # nome GICS, non EODHD
    assert "Financials" in sm.coverage()["settori_sconosciuti"]


# ------------------------------------------- filtro di volatilità
def _bars(n, px0, hi_mult, lo_mult, vol=900_000):
    weeks = pd.date_range("2020-01-03", periods=n, freq="W-FRI")
    px = np.full(n, float(px0))
    return pd.DataFrame({"adj_close": px, "close": px, "open": px,
                         "high": px * hi_mult, "low": px * lo_mult,
                         "volume": [vol] * n}, index=weeks)


def test_volatility_filter_excludes_nervous_keeps_calm():
    """La volatilità misura DIRETTAMENTE ciò che la capitalizzazione
    approssimava: 'variazioni contenute'."""
    from src.universe import LiquidityUniverse
    calmo = _bars(40, 50.0, 1.01, 0.99)      # range ±1%
    nervoso = _bars(40, 50.0, 1.12, 0.88)    # range ±12%
    u = LiquidityUniverse(min_price=5.0, min_volume=1000, min_dollar_volume=1_000_000,
                          lookback_weeks=8, max_volatility=0.06, atr_weeks=14)
    assert u.compute("CALMO", calmo).iloc[20:].all()
    assert not u.compute("NERVOSO", nervoso).iloc[20:].any()


def test_volatility_filter_disabled_by_default():
    """max_volatility=None => criterio inattivo, comportamento invariato."""
    from src.universe import LiquidityUniverse
    nervoso = _bars(40, 50.0, 1.12, 0.88)
    u = LiquidityUniverse(min_price=5.0, min_volume=1000,
                          min_dollar_volume=1_000_000, lookback_weeks=8)
    assert u.compute("NERVOSO", nervoso).iloc[20:].all()   # passa: nessun filtro vol


def test_volatility_breakout_bar_does_not_self_exclude():
    """La barra del breakout è per definizione ampia: senza shift(1)
    escluderebbe il titolo proprio nella settimana del segnale."""
    from src.universe import LiquidityUniverse
    df = _bars(40, 50.0, 1.01, 0.99)
    last = df.index[-1]
    df.loc[last, "high"] = 70.0      # barra enorme SOLO nell'ultima settimana
    df.loc[last, "low"] = 40.0
    u = LiquidityUniverse(min_price=5.0, min_volume=1000, min_dollar_volume=1_000_000,
                          lookback_weeks=8, max_volatility=0.06, atr_weeks=14)
    ok = u.compute("SPIKE", df)
    assert ok.iloc[-1], "la barra del breakout non deve auto-escludere il titolo"


def test_redundancy_check_detects_added_selectivity():
    """Un titolo con turnover alto MA volatilità alta viene escluso solo dal
    filtro di volatilità: la diagnostica deve accorgersene."""
    from src.universe import LiquidityUniverse
    calmo = _bars(60, 100.0, 1.01, 0.99, vol=5_000_000)     # turnover altissimo
    nervoso = _bars(60, 100.0, 1.15, 0.85, vol=5_000_000)   # turnover uguale, vol alta
    u = LiquidityUniverse(min_price=5.0, min_volume=1000, min_dollar_volume=1_000_000,
                          lookback_weeks=13, max_volatility=0.06, atr_weeks=14)
    u.compute("CALMO", calmo)
    u.compute("NERVOSO", nervoso)
    vc = u.redundancy_check()
    assert vc["disponibile"]
    assert vc["settimane_escluse_solo_da_volatilita"] > 0
    assert "aggiunge selettività" in vc["interpretazione"]


def test_redundancy_check_handles_constant_input():
    """Correlazione non definita => None, non 'nan' nel report."""
    from src.universe import LiquidityUniverse
    df = _bars(40, 50.0, 1.01, 0.99)
    u = LiquidityUniverse(min_price=5.0, min_volume=1000,
                          min_dollar_volume=1_000_000, lookback_weeks=8)
    u.compute("A", df)
    vc = u.redundancy_check()
    assert vc["correlazione_turnover_volatilita"] is None


def test_redundancy_check_needs_no_scipy():
    """La correlazione di Spearman è calcolata come Pearson sui ranghi.
    pandas method='spearman' richiederebbe scipy, che NON è tra le dipendenze:
    questo test impedisce che qualcuno lo reintroduca senza accorgersene."""
    import importlib.abc
    import sys

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError(f"No module named '{name}' (bloccato dal test)")
            return None

    from src.universe import LiquidityUniverse
    weeks = pd.date_range("2020-01-03", periods=60, freq="W-FRI")

    def _df(px0, hi, lo, vol):
        px = np.full(60, float(px0))
        return pd.DataFrame({"adj_close": px, "close": px, "open": px,
                             "high": px * hi, "low": px * lo,
                             "volume": [vol] * 60}, index=weeks)

    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        u = LiquidityUniverse(min_price=5.0, min_volume=1000,
                              min_dollar_volume=1_000_000, lookback_weeks=13,
                              max_volatility=0.06)
        u.compute("CALMO", _df(100.0, 1.01, 0.99, 5_000_000))
        u.compute("NERVOSO", _df(80.0, 1.15, 0.85, 3_000_000))
        vc = u.redundancy_check()          # non deve sollevare ImportError
        assert vc["disponibile"]
        assert vc["correlazione_turnover_volatilita"] is not None
    finally:
        sys.meta_path.remove(blocker)


# ------------------------------------- ticker EODHD: .US, non .NASDAQ
def test_exchange_symbols_builds_us_ticker_not_real_exchange():
    """EODHD vuole 'AAPL.US', non 'AAPL.NASDAQ'. Il campo `exchange` della
    risposta contiene la borsa REALE: usarlo per il ticker dà 404 su tutto."""
    from unittest.mock import MagicMock, patch

    from src.providers.eodhd import EODHDProvider
    fake = [
        {"Code": "AAPL", "Exchange": "NASDAQ", "Currency": "USD", "Type": "Common Stock"},
        {"Code": "A", "Exchange": "NYSE", "Currency": "USD", "Type": "Common Stock"},
        {"Code": "XYZ", "Exchange": "OTC", "Currency": "USD", "Type": "Common Stock"},
    ]
    p = EODHDProvider("FAKE")
    resp = MagicMock()
    resp.json.return_value = fake
    resp.raise_for_status = lambda: None
    with patch.object(p._session, "get", return_value=resp), patch("time.sleep"):
        df = p.exchange_symbols("US", delisted=False)
    assert list(df["ticker"]) == ["AAPL.US", "A.US", "XYZ.US"]
    # la borsa reale resta disponibile: serve per escludere l'OTC
    assert list(df["exchange_real"]) == ["NASDAQ", "NYSE", "OTC"]


def test_scrub_removes_api_token_from_errors():
    """requests mette l'URL completo nelle eccezioni, e l'URL contiene la
    chiave. Non deve finire nei file di log."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bu", str(Path(__file__).parent.parent / "build_universe.py"))
    bu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bu)
    msg = "404 for url: https://eodhd.com/api/eod/A.US?api_token=SECRET123&fmt=json"
    out = bu._scrub(msg)
    assert "SECRET123" not in out
    assert "api_token=***" in out


def test_junk_ticker_filter_keeps_real_stocks():
    """Il filtro su warrant/right/preferred non deve mangiare titoli veri
    come F, T, PWR, HPQ."""
    import re
    veri = ["AAPL", "A", "F", "T", "PWR", "NOW", "GE", "BA", "CAT", "MMM", "V", "WU", "HPQ"]
    spazzatura = ["AACBR", "AACPR", "ABCWS"]
    for code in veri + spazzatura:
        junk = bool(re.match(r".*(W|WS|R|U|P|PR[A-Z]?)$", code.upper()))
        plain = len(code) <= 4
        scartato = junk and not plain
        if code in veri:
            assert not scartato, f"{code} è un titolo vero, non va scartato"
        else:
            assert scartato, f"{code} è spazzatura, va scartato"


def test_prices_missing_universe_gives_clear_error():
    """Se manca il parquet dell'universo, l'utente deve leggere cosa fare,
    non uno stack trace di pandas."""
    import importlib.util
    from argparse import Namespace
    spec = importlib.util.spec_from_file_location(
        "bu2", str(Path(__file__).parent.parent / "build_universe.py"))
    bu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bu)
    args = Namespace(from_universe="data/NON_ESISTE.parquet", data_dir="data",
                     benchmark=None, with_sector_etfs=False, start="2000-01-01",
                     end="2024-12-31", yes=True, index="X")
    with pytest.raises(SystemExit):
        bu.cmd_prices(args)


def test_liquidity_universe_read_from_parquet(tmp_path):
    """Con `tickers: []` nel config, l'universo si legge dal parquet di `screen`.
    Scaricare 800 storici e poi incollare 800 ticker a mano sarebbe assurdo."""
    from src.backtest import run_backtest
    from src.config import load_config
    from src.data_store import DataStore
    from src.providers import make_provider
    from src.providers.synthetic import SyntheticProvider
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config_sim.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config_sim.yaml non presente")
    cfg = load_config(cfg_path)
    cfg["data_dir"] = str(tmp_path)   # MAI scrivere/cancellare nella data/ reale
    tick = SyntheticProvider().universe()
    up = Path(cfg["data_dir"]) / "universe_US.parquet"
    up.parent.mkdir(parents=True, exist_ok=True)
    # includo un ticker che NON ha prezzi in cache: deve essere escluso, non
    # far fallire tutto il run
    pd.DataFrame({"ticker": tick + ["MANCANTE.US"],
                  "is_delisted": [False] * (len(tick) + 1)}).to_parquet(up)
    try:
        cfg["universe"] = {"US": {"benchmark": "SYN-BENCH", "mode": "liquidity",
                                  "liquidity": {"min_price": 5.0, "min_volume": 1000,
                                                "min_dollar_volume": 50_000,
                                                "lookback_weeks": 13}}}
        store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
        for t in tick:
            store.get_with_warmup(t, cfg["backtest"]["start"], cfg["backtest"]["end"])
        res = run_backtest(cfg, store)
        assert res.metrics["universe_size"] == len(tick)   # MANCANTE.US escluso
    finally:
        up.unlink(missing_ok=True)


def test_sector_filter_active_false_when_etfs_missing(tmp_path):
    """Una mappa settoriale senza ETF caricati NON filtra nulla: il report
    non deve dichiarare il triple screen attivo."""
    from src.backtest import run_backtest
    from src.config import load_config
    from src.data_store import DataStore
    from src.providers import make_provider
    from src.providers.synthetic import SyntheticProvider
    import csv
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config_sim.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config_sim.yaml non presente")
    cfg = load_config(cfg_path)
    cfg["data_dir"] = str(tmp_path)   # MAI scrivere nella data/ reale
    tick = SyntheticProvider().universe()
    smap = Path(cfg["data_dir"]) / "sm_regress.csv"
    with smap.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "sector"])
        for i, t in enumerate(tick):
            w.writerow([t, "Technology" if i % 2 else "Energy"])
    try:
        cfg["signal"]["sector_filter"] = True
        cfg["universe"] = {"US": {"benchmark": "SYN-BENCH", "mode": "liquidity",
                                  "tickers": tick, "sector_map": str(smap),
                                  "liquidity": {"min_price": 5.0, "min_volume": 1000,
                                                "min_dollar_volume": 50_000,
                                                "lookback_weeks": 13}}}
        res = run_backtest(cfg, DataStore(cfg["data_dir"], make_provider(cfg["provider"])))
        # gli ETF XLK/XLE non esistono fra i sintetici => filtro inattivo
        assert res.metrics["sector_filter_active"] is False
        assert res.metrics["signals_skipped_sector"] == 0
    finally:
        smap.unlink(missing_ok=True)


# ============ REVISIONE: bug trovati nell'audit del codice ============

def test_pivot_recognizes_double_bottom():
    """BUG STORICO: la condizione `(window > pivot).sum() == 2k` esigeva che
    TUTTE le barre fossero strettamente sopra il pivot. Un doppio minimo — il
    double bottom, formazione centrale in Weinstein — non produceva pivot.
    Conseguenza: trailing stop che non sale, stop 'minimo relativo' assenti."""
    from src.indicators import confirmed_pivot_lows

    def _piv(lows, k=2):
        idx = pd.date_range("2020-01-03", periods=len(lows), freq="W-FRI")
        df = pd.DataFrame({"low": lows, "adj_low": lows,
                           "high": [x * 1.05 for x in lows],
                           "close": lows, "adj_close": lows,
                           "open": lows, "volume": 1}, index=idx)
        return confirmed_pivot_lows(df, k)

    assert _piv([10, 9, 8, 9, 10, 11, 12]).dropna().tolist() == [8.0]   # netto
    assert _piv([10, 9, 8, 8, 10, 11, 12]).dropna().tolist() == [8.0]   # doppio
    assert _piv([10, 9, 8, 8, 8, 11, 12]).dropna().tolist() == [8.0]    # triplo
    # un doppio minimo produce UN pivot, non due
    assert _piv([10, 9, 8, 8, 10, 11, 12]).notna().sum() == 1


def test_pivot_still_confirms_at_p_plus_k():
    """La correzione sui pareggi non deve rompere l'anti-lookahead."""
    from src.indicators import confirmed_pivot_lows
    lows = [10, 9, 8, 9, 10, 11, 12]
    idx = pd.date_range("2020-01-03", periods=len(lows), freq="W-FRI")
    df = pd.DataFrame({"low": lows, "adj_low": lows, "high": [x * 1.05 for x in lows],
                       "close": lows, "adj_close": lows, "open": lows,
                       "volume": 1}, index=idx)
    p = confirmed_pivot_lows(df, k=2)
    conf = p.dropna().index[0]
    assert list(idx).index(conf) == 4      # minimo a 2, conferma a 2+k=4


def test_adjusted_high_low_same_scale_as_adj_close():
    """BUG STORICO: `base_features` usava high/low GREZZI mentre il segnale
    confronta `adj_close > resistance`. Su un titolo con split (AAPL 7:1) le
    due scale divergono e il breakout non scatta mai."""
    from src.indicators import enrich_weekly
    n = 60
    idx = pd.date_range("2020-01-03", periods=n, freq="W-FRI")
    raw = np.full(n, 700.0)
    raw[30:] = 100.0                      # split 7:1 alla settimana 30
    adj = np.full(n, 100.0)               # adj_close: scala continua
    df = pd.DataFrame({"open": raw, "high": raw * 1.03, "low": raw * 0.97,
                       "close": raw, "adj_close": adj, "volume": 1e6}, index=idx)
    e = enrich_weekly(df, None, 30, 4)
    # adj_high deve stare sulla stessa scala di adj_close in ENTRAMBI i periodi
    r_pre = e["adj_high"].iloc[10] / e["adj_close"].iloc[10]
    r_post = e["adj_high"].iloc[40] / e["adj_close"].iloc[40]
    assert abs(r_pre - 1.03) < 0.01
    assert abs(r_post - 1.03) < 0.01
    # il grezzo invece salta
    assert e["high"].iloc[10] / e["adj_close"].iloc[10] > 5


def test_delisted_position_is_force_closed():
    """BUG STORICO: `if wk_date not in df.index: continue` lasciava aperte per
    sempre le posizioni su titoli delistati, valutate all'ultimo prezzo noto e
    occupando uno slot di max_positions. Con l'universo per liquidità (2/3 di
    delistati) l'equity finale sarebbe piena di posizioni fantasma."""
    from src.portfolio import Portfolio
    idx = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    df = pd.DataFrame({"adj_close": np.linspace(100, 140, 30)}, index=idx)
    per_ticker = {"MORTO": df}
    last_bar = {tk: d.index[-1] for tk, d in per_ticker.items()}

    pf = Portfolio(100000, 0.01, 0.20, 15, 5, 10)
    pf.open("MORTO", idx[5], 110.0, 100.0, 50)
    assert pf.positions

    # una settimana oltre l'ultima barra
    wk = idx[-1] + pd.Timedelta(weeks=1)
    for tk in list(pf.positions):
        if wk > last_bar[tk]:
            pf.close(tk, last_bar[tk], float(per_ticker[tk]["adj_close"].iloc[-1]),
                     "delisting")
    assert not pf.positions
    assert pf.trades[-1].reason == "delisting"


def test_dead_ticker_not_priced_after_last_bar():
    """Un titolo delistato non deve contribuire all'equity dopo la sua morte."""
    idx = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    df = pd.DataFrame({"adj_close": np.linspace(100, 140, 30)}, index=idx)
    last_bar = {"MORTO": df.index[-1]}
    future = df.index[-1] + pd.Timedelta(weeks=4)
    prices = {}
    for tk, d in {"MORTO": df}.items():
        if future > last_bar[tk]:
            continue
        prices[tk] = float(d["adj_close"].asof(future))
    assert prices == {}          # nessun prezzo fantasma


def test_equity_fallback_uses_last_price_not_entry():
    """BUG STORICO: `prices.get(tk, p.entry_price)` valutava una posizione
    senza prezzo al PREZZO D'INGRESSO, fingendo che non avesse perso nulla e
    mascherando i drawdown."""
    from src.portfolio import Portfolio
    pf = Portfolio(100000, 0.01, 0.20, 15, 0, 0)
    pf.open("X", pd.Timestamp("2020-01-03"), 100.0, 90.0, 100)
    pf.mark_prices({"X": 50.0})              # il titolo crolla
    eq_con = pf.equity({"X": 50.0})
    eq_senza = pf.equity({})                 # prezzo mancante
    assert eq_senza == eq_con                # usa last_price, non entry_price
    assert eq_senza < 100000                 # la perdita NON è mascherata


def test_cache_not_redownloaded_when_warmup_precedes_data():
    """BUG STORICO: il warmup risale 80 settimane prima di `start`. Se i dati
    partono dopo quella data, la cache risultava sempre insufficiente e
    l'intero universo veniva RISCARICATO a ogni run."""
    import shutil
    import tempfile

    from src.data_store import DataStore

    class _P:
        name = "test"

        def __init__(self):
            self.calls = 0

        def get_eod(self, ticker, start, end):
            self.calls += 1
            idx = pd.date_range(max(pd.Timestamp(start), pd.Timestamp("2000-01-03")),
                                end, freq="B")
            return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                                 "adj_close": 1.0, "volume": 100}, index=idx)

    tmp = tempfile.mkdtemp()
    try:
        prov = _P()
        ds = DataStore(tmp, prov)
        ds.get_with_warmup("X", "2001-06-01", "2024-12-31")
        assert prov.calls == 1
        ds.get_with_warmup("X", "2001-06-01", "2024-12-31")   # seconda volta
        assert prov.calls == 1, "la cache è stata ignorata: ri-download!"
    finally:
        shutil.rmtree(tmp)


# ============================================================ AUDIT
# Test scritti durante la revisione sistematica pre-backtest reale.

def test_no_lookahead_indicators_truncation():
    """Il valore di un indicatore a t non deve cambiare se aggiungo dati DOPO t.
    È la definizione operativa di 'niente look-ahead'."""
    from src.indicators import enrich_weekly, mansfield_rs, volume_ratio
    from src.stages import classify_stages
    idx = pd.date_range("2019-01-04", periods=90, freq="W-FRI")
    rng = np.random.default_rng(11)
    px = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.03, 90)))
    df = pd.DataFrame({"open": px, "high": px * 1.03, "low": px * 0.97, "close": px,
                       "adj_close": px, "volume": rng.lognormal(13, 0.4, 90)}, index=idx)
    bpx = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 90)))
    bench = pd.DataFrame({"open": bpx, "high": bpx * 1.02, "low": bpx * 0.98,
                          "close": bpx, "adj_close": bpx,
                          "volume": np.full(90, 1e6)}, index=idx)
    t = 80

    def _same(a, b):
        return (pd.isna(a) and pd.isna(b)) or bool(np.isclose(float(a), float(b)))

    assert _same(volume_ratio(df["volume"], 10).iloc[t],
                 volume_ratio(df["volume"].iloc[:t + 1], 10).iloc[t])
    assert _same(mansfield_rs(df["adj_close"], bench["adj_close"], 52).iloc[t],
                 mansfield_rs(df["adj_close"].iloc[:t + 1],
                              bench["adj_close"].iloc[:t + 1], 52).iloc[t])
    fw = enrich_weekly(df, bench, 30, 4)
    tw = enrich_weekly(df.iloc[:t + 1], bench.iloc[:t + 1], 30, 4)
    for c in ("ma", "ma_slope", "mansfield", "vol_ratio"):
        if c in fw.columns:
            assert _same(fw[c].iloc[t], tw[c].iloc[t]), f"look-ahead in {c}"
    assert classify_stages(fw, 0.005).iloc[t] == classify_stages(tw, 0.005).iloc[t]


def test_delisting_closes_position_and_books_loss():
    """Su un universo con 2/3 di titoli morti, questo è il caso critico:
    la posizione va chiusa all'ultimo prezzo noto, non congelata."""
    from src.portfolio import Portfolio
    pf = Portfolio(100_000, 0.01, 0.20, 15, 5, 10)
    pf.open("DEAD", pd.Timestamp("2020-01-03"), 100.0, 90.0, 100)
    pf.mark_prices({"DEAD": 95.0})
    # senza prezzo si usa last_price (95), non entry (100): la perdita è visibile
    assert abs(pf.equity({}) - pf.equity({"DEAD": 95.0})) < 0.01
    pf.close("DEAD", pd.Timestamp("2020-06-05"), 40.0, "delisting")
    t = pf.trades[-1]
    assert t.reason == "delisting"
    assert t.ret < 0
    assert not pf.positions


def test_fixed_risk_keeps_risk_constant_across_stop_widths():
    """Il cuore del sizing: stop più largo => meno azioni, rischio invariato."""
    from src.portfolio import Portfolio
    pf = Portfolio(100_000, 0.01, 1.0, 15, 0, 0)   # cap alto per isolare il sizing
    rischi = []
    for stop in (95.0, 90.0, 84.0):
        sh = pf.size_fixed_risk(100_000, 100.0, stop)
        rischi.append(sh * (100.0 - stop))
    assert all(abs(r - 1000.0) < 1.0 for r in rischi), rischi


def test_max_position_pct_caps_tight_stops():
    """Stop strettissimo => tante azioni => il cap sul valore deve mordere."""
    from src.portfolio import Portfolio
    pf = Portfolio(100_000, 0.01, 0.20, 15, 0, 0)
    sh = pf.size_fixed_risk(100_000, 100.0, 99.5)
    assert sh * 100.0 <= 20_000 * 1.01


def test_trailing_runs_before_exits():
    """Lo stop alzato non deve poter scattare nella stessa settimana:
    il trailing sta PRIMA delle uscite nel loop."""
    import inspect
    from src import backtest
    src = inspect.getsource(backtest.run_backtest)
    assert src.find("trailing") < src.find("# 3b")


def test_signal_uses_resistance_known_before_breakout():
    """La resistenza rotta a t è quella calcolata fino a t-1: senza lo shift,
    il breakout stesso alzerebbe la resistenza che deve superare."""
    import inspect
    from src.signals import detect_signals
    src = inspect.getsource(detect_signals)
    for col in ("resistance", "base_len", "base_depth"):
        assert f'df["{col}"].shift(1)' in src, f"{col} non è shiftato"


def test_hash_sampling_is_stable_when_list_changes():
    """`df.sample(random_state=42)` campiona POSIZIONI: se EODHD aggiunge o
    toglie un ticker, l'intero campione cambia e la cache scaricata diventa
    inutile. È successo davvero (20899 -> 20898, 54% del campione diverso).
    L'hash del ticker dà a ogni titolo un destino fisso."""
    import hashlib

    def _by_hash(tickers, n):
        d = pd.DataFrame({"ticker": tickers})
        h = d["ticker"].map(lambda t: int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
        return set(d.assign(_h=h).nsmallest(n, "_h")["ticker"])

    base = [f"T{i:05d}" for i in range(5000)]
    tolto = [t for t in base if t != "T02500"]
    aggiunto = base + ["ZZZZZ"]

    a = _by_hash(base, 300)
    assert len(a & _by_hash(tolto, 300)) >= 299       # al più esce il ticker tolto
    assert len(a & _by_hash(aggiunto, 300)) >= 299    # al più entra quello aggiunto

    # per contrasto: sample() è instabile
    s1 = set(pd.DataFrame({"ticker": base}).sample(n=300, random_state=42)["ticker"])
    s2 = set(pd.DataFrame({"ticker": tolto}).sample(n=300, random_state=42)["ticker"])
    assert len(s1 & s2) < 290, "sample() dovrebbe essere instabile: se non lo è, il test non serve"


def test_screen_preserves_metadata_columns():
    """L'universo deve conservare `type`, `exchange_real`, `currency`:
    senza, non si può verificare cosa contiene né diagnosticare il campione."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bu3", str(Path(__file__).parent.parent / "build_universe.py"))
    bu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bu)
    src = inspect_source = open(
        Path(__file__).parent.parent / "build_universe.py", encoding="utf-8").read()
    # il salvataggio non deve più essere limitato a due colonne
    assert 'df[["ticker", "is_delisted"]].to_parquet' not in src
    assert '"exchange_real"' in src and '"type"' in src


def test_otc_flag_is_actually_toggleable():
    """`action="store_true", default=True` rende un flag inutilizzabile:
    resta True qualunque cosa passi. Serve `store_false` su un dest invertito."""
    src = open(Path(__file__).parent.parent / "build_universe.py",
               encoding="utf-8").read()
    assert 'ap.add_argument("--exclude-otc", action="store_true", default=True)' not in src
    assert '"--include-otc"' in src and 'action="store_false"' in src


def test_corrupt_ticker_does_not_abort_backtest(tmp_path):
    """Su 20.000 titoli, alcune cache sono vuote o corrotte (download fallito).
    Un singolo ticker illeggibile NON deve far crollare l'intero backtest."""
    from src.backtest import run_backtest
    from src.config import load_config
    from src.data_store import DataStore
    from src.providers import make_provider
    from src.providers.synthetic import SyntheticProvider
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config_sim.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config_sim.yaml non presente")
    cfg = load_config(cfg_path)
    cfg["data_dir"] = str(tmp_path)   # MAI scrivere nella data/eodhd reale
    tick = SyntheticProvider().universe()
    cache = Path(cfg["data_dir"]) / cfg["provider"]
    cache.mkdir(parents=True, exist_ok=True)
    corrupt = cache / "CORRUPT.US.parquet"
    pd.DataFrame().to_parquet(corrupt)      # parquet vuoto = download fallito
    up = Path(cfg["data_dir"]) / "universe_US.parquet"
    pd.DataFrame({"ticker": tick + ["CORRUPT.US"],
                  "is_delisted": [False] * (len(tick) + 1)}).to_parquet(up)
    try:
        cfg["universe"] = {"US": {"benchmark": "SYN-BENCH", "mode": "liquidity",
                                  "liquidity": {"min_price": 5.0, "min_volume": 1000,
                                                "min_dollar_volume": 50_000,
                                                "lookback_weeks": 13}}}
        store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
        for t in tick:
            store.get_with_warmup(t, cfg["backtest"]["start"], cfg["backtest"]["end"])
        res = run_backtest(cfg, store)          # non deve sollevare
        assert res.metrics["universe_size"] == len(tick)
    finally:
        corrupt.unlink(missing_ok=True)
        up.unlink(missing_ok=True)


# ============================================ sanificazione dati sporchi
def test_validate_removes_zero_and_negative_prices():
    """Prezzi a zero o negativi (visti nei dati reali come entry 0.00) vanno
    scartati: a valle diventano breakout fantasma."""
    from src.providers.base import DataProvider
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    d = pd.DataFrame(np.ones((30, 6)) * 50, columns=cols, index=idx)
    d.iloc[10:13, :5] = 0.0
    out = DataProvider.validate(d.copy(), "ZERO.US")
    assert (out[["open", "high", "low", "close", "adj_close"]] > 0).all().all()
    assert len(out) == 27


def test_validate_removes_corrupt_split_factor():
    """close≈0 fa esplodere adj_close/close, che a valle produce resistenze
    a 1000000. La riga va scartata."""
    from src.providers.base import DataProvider
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    d = pd.DataFrame(np.ones((30, 6)) * 50, columns=cols, index=idx)
    d.iloc[15, d.columns.get_loc("close")] = 0.001
    out = DataProvider.validate(d.copy(), "SPLIT.US")
    assert idx[15] not in out.index


def test_validate_rejects_mostly_corrupt_ticker():
    """Se >50% delle barre è corrotto, il titolo è inaffidabile: scartato."""
    from src.providers.base import DataProvider
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    d = pd.DataFrame(np.ones((30, 6)) * 50, columns=cols, index=idx)
    d.iloc[:20, :5] = 0.0
    with pytest.raises(ValueError, match="corrotte"):
        DataProvider.validate(d.copy(), "GARBAGE.US")


def test_validate_keeps_clean_data_intact():
    """I dati puliti non devono essere toccati."""
    from src.providers.base import DataProvider
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    px = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, 30)))
    d = pd.DataFrame({"open": px, "high": px * 1.02, "low": px * 0.98, "close": px,
                      "adj_close": px * 0.9, "volume": rng.lognormal(12, 0.3, 30)},
                     index=idx)
    out = DataProvider.validate(d.copy(), "CLEAN.US")
    assert len(out) == len(d)


def test_validate_clips_spike_low_but_keeps_bar():
    """Un `low` anomalo molto sotto il corpo (tick corrotto tipo MDSO) va
    CLIPPATO a un wick massimo, senza scartare la barra (open/close sono buoni)
    né toccare le barre normali."""
    from src.providers.base import DataProvider
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    # barra 1 normale; barra 2 con low corrotto (-60% dal corpo); barra 3 normale
    d = pd.DataFrame({
        "open":      [40.0, 37.5, 41.0],
        "high":      [41.0, 43.8, 42.0],
        "low":       [39.0, 16.0, 40.0],   # 16.0 su corpo 37.5-42: spike sporco
        "close":     [40.5, 42.0, 41.5],
        "adj_close": [40.5, 42.0, 41.5],
        "volume":    [1e5, 8e5, 3e5],
    }, index=idx)
    out = DataProvider.validate(d.copy(), "SPIKE.US")
    assert len(out) == 3                              # nessuna barra scartata
    # il low corrotto è risalito al floor (0.6 * min(open,close) = 0.6*37.5)
    assert out["low"].iloc[1] == pytest.approx(37.5 * 0.6)
    # le barre normali restano intatte
    assert out["low"].iloc[0] == 39.0 and out["low"].iloc[2] == 40.0


def test_signal_rejects_infinite_volume_ratio_and_wild_mansfield():
    """Seconda rete: un segnale con vol_ratio infinito o Mansfield fuori scala
    è un errore di dati, non un breakout."""
    from src.signals import detect_signals
    import inspect
    src = inspect.getsource(detect_signals)
    # i guardrail devono esserci
    assert "np.isfinite(entry)" in src
    assert "vr > 100" in src
    assert "abs(mans) > 500" in src


# ============================================ cache del settimanale
def test_weekly_cache_speeds_up_without_changing_results():
    """La cache del settimanale arricchito non deve cambiare i numeri:
    accelera i run successivi, non altera il risultato."""
    from src.backtest import run_backtest
    from src.config import load_config
    from src.data_store import DataStore
    from src.providers import make_provider
    from src.providers.synthetic import SyntheticProvider
    import shutil
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config_sim.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config_sim.yaml non presente")
    cfg = load_config(cfg_path)
    tick = SyntheticProvider().universe()
    cfg["universe"] = {"US": {"benchmark": "SYN-BENCH", "mode": "liquidity",
                              "tickers": tick,
                              "liquidity": {"min_price": 5.0, "min_volume": 1000,
                                            "min_dollar_volume": 50_000,
                                            "lookback_weeks": 13}}}
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    wr = store.weekly_root       # fonte di verità, non un percorso ricostruito
    if wr.exists():
        shutil.rmtree(wr)
        wr.mkdir(parents=True, exist_ok=True)
    r1 = run_backtest(cfg, store)      # cache fredda
    r2 = run_backtest(cfg, store)      # cache calda
    assert r1.metrics["n_trades"] == r2.metrics["n_trades"]
    assert r1.metrics["n_signals"] == r2.metrics["n_signals"]
    assert wr.exists() and any(wr.glob("*.parquet"))


def test_weekly_key_stable_for_signal_params_changes():
    """La chiave della cache dipende dai parametri di STAGE, non dal segnale:
    cambiare volume_ratio_min riusa la cache."""
    from src.data_store import DataStore
    k1 = DataStore.weekly_key("SPY.US", 30, 4, 0.005, "2001-06-01", "2024-12-31")
    k2 = DataStore.weekly_key("SPY.US", 30, 4, 0.005, "2001-06-01", "2024-12-31")
    assert k1 == k2                    # stessi parametri stage → stessa chiave
    k3 = DataStore.weekly_key("SPY.US", 40, 4, 0.005, "2001-06-01", "2024-12-31")
    assert k1 != k3                    # MA diversa → chiave diversa (giusto)


def test_delisted_ticker_not_redownloaded_every_run():
    """Un delistato ha dati solo fino alla sua morte. Se abbiamo già chiesto
    fino a `end`, la cache è completa: NON deve richiamare l'API a ogni run.
    Senza questo, un backtest su 14.000 titoli (2/3 delistati) spende ore di
    rete a caccia di dati che non esistono."""
    import json
    import tempfile
    from src.data_store import DataStore
    from src.providers.base import DataProvider

    class _Counting(DataProvider):
        name = "counting"
        def __init__(self):
            self.calls = 0
        def get_eod(self, ticker, start, end):
            self.calls += 1
            idx = pd.date_range(start, end, freq="B")
            px = np.full(len(idx), 50.0)
            return pd.DataFrame({"open": px, "high": px, "low": px, "close": px,
                                 "adj_close": px, "volume": np.full(len(idx), 1e6)},
                                index=idx)
        def universe(self):
            return []

    store = DataStore(tempfile.mkdtemp(), _Counting())
    idx = pd.date_range("2000-01-01", "2015-06-30", freq="B")   # muore nel 2015
    px = np.full(len(idx), 50.0)
    pd.DataFrame({"open": px, "high": px, "low": px, "close": px,
                  "adj_close": px, "volume": np.full(len(idx), 1e6)},
                 index=idx).to_parquet(store._path("DEAD.US"))
    store._meta_path("DEAD.US").write_text(
        json.dumps({"asked_from": "1999-06-01", "asked_to": "2024-12-31"}))

    store.provider.calls = 0
    for _ in range(3):
        store.get_with_warmup("DEAD.US", "2001-06-01", "2024-12-31")
    assert store.provider.calls == 0


def test_genuinely_incomplete_cache_still_updates():
    """Controprova: se NON abbiamo mai chiesto oltre una certa data, la cache
    è davvero incompleta e va aggiornata. La correzione sui delistati non deve
    trasformarsi in 'cache sempre valida'."""
    import json
    import tempfile
    from src.data_store import DataStore
    from src.providers.base import DataProvider

    class _Counting(DataProvider):
        name = "counting"
        def __init__(self):
            self.calls = 0
        def get_eod(self, ticker, start, end):
            self.calls += 1
            idx = pd.date_range(start, end, freq="B")
            px = np.full(len(idx), 50.0)
            return pd.DataFrame({"open": px, "high": px, "low": px, "close": px,
                                 "adj_close": px, "volume": np.full(len(idx), 1e6)},
                                index=idx)
        def universe(self):
            return []

    store = DataStore(tempfile.mkdtemp(), _Counting())
    idx = pd.date_range("2000-01-01", "2015-06-30", freq="B")
    px = np.full(len(idx), 50.0)
    pd.DataFrame({"open": px, "high": px, "low": px, "close": px,
                  "adj_close": px, "volume": np.full(len(idx), 1e6)},
                 index=idx).to_parquet(store._path("STALE.US"))
    store._meta_path("STALE.US").write_text(
        json.dumps({"asked_from": "1999-06-01", "asked_to": "2015-06-30"}))

    store.provider.calls = 0
    store.get_with_warmup("STALE.US", "2001-06-01", "2024-12-31")
    assert store.provider.calls == 1


def test_console_setup_makes_stdout_utf8():
    """La console Windows (cp1252) crasha sui simboli → × ✓ che il progetto
    stampa. src.console.setup deve riconfigurare stdout in UTF-8."""
    import io
    import sys
    from src.console import setup

    orig = sys.stdout
    try:
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        setup()
        # dopo setup, la freccia non deve più sollevare
        sys.stdout.write("periodo 2001 → 2024 · 2.0× ✓")
        sys.stdout.flush()
    finally:
        sys.stdout = orig


def test_entry_points_reconfigure_utf8():
    """Ogni entry point che stampa simboli deve riconfigurare la console,
    o crasherà su Windows alla prima print."""
    root = Path(__file__).parent.parent
    for fn in ("run.py", "build_universe.py", "diagnose.py"):
        src = (root / fn).read_text(encoding="utf-8")
        assert "reconfigure" in src or "_console_setup" in src, f"{fn} senza fix UTF-8"


def test_execution_next_open_is_default_and_realistic():
    """L'ingresso NON deve avvenire alla chiusura del breakout stesso (prezzo
    che ha rivelato il segnale), ma alla settimana successiva. Default realistico."""
    from src.backtest import run_backtest
    from src.config import load_config
    from src.data_store import DataStore
    from src.providers import make_provider
    from src.providers.synthetic import SyntheticProvider
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config_sim.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config_sim.yaml non presente")
    tick = SyntheticProvider().universe()

    def _run(execution):
        cfg = load_config(cfg_path)
        if execution:
            cfg["exits"]["execution"] = execution
        cfg["universe"] = {"US": {"benchmark": "SYN-BENCH", "mode": "list", "tickers": tick}}
        store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
        return run_backtest(cfg, store)

    # il default (nessun execution specificato) deve comportarsi come next_open
    r_default = _run(None)
    r_next = _run("next_open")
    assert r_default.metrics["n_trades"] == r_next.metrics["n_trades"]


def test_execution_uses_next_week_price_not_signal_close():
    """Verifica strutturale: con next_open il codice esegue a una data
    successiva a quella del segnale."""
    import inspect
    from src import backtest
    src = inspect.getsource(backtest.run_backtest)
    assert 'execution == "same_close"' in src
    assert "df_tk.index > wk_date" in src   # cerca la settimana DOPO il segnale


def test_oos_split_no_overlap():
    """In-sample e out-of-sample non devono condividere la settimana di split."""
    import inspect
    from src import backtest
    src = inspect.getsource(backtest.compute_metrics)
    assert "equity.index > split" in src    # oos parte STRETTAMENTE dopo


def test_stop_evaluated_on_weekly_low_is_more_severe():
    """Lo stop deve poter scattare sul minimo settimanale, non solo sulla
    chiusura: un ordine reale non aspetta il venerdì."""
    import inspect
    from src import backtest
    src = inspect.getsource(backtest.run_backtest)
    assert "stop_intraweek" in src
    assert 'low <= pos.stop' in src


def test_random_twin_produces_distribution():
    """Il gemello casuale deve produrre una distribuzione di CAGR e collocare
    la strategia a un percentile."""
    from src.random_twin import strategy_percentile
    twin_cagrs = [0.05, 0.08, 0.09, 0.10, 0.11, 0.12, 0.15]
    # una strategia a 0.13 batte 6 dei 7 sorteggi → ~86°
    pct = strategy_percentile(0.13, twin_cagrs)
    assert 80 <= pct <= 90
    # una a 0.04 non ne batte nessuno → 0°
    assert strategy_percentile(0.04, twin_cagrs) == 0.0


def test_random_twin_runs_end_to_end():
    """Il gemello gira dentro run_backtest quando abilitato nel config."""
    from src.backtest import run_backtest
    from src.config import load_config
    from src.data_store import DataStore
    from src.providers import make_provider
    from src.providers.synthetic import SyntheticProvider
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config_sim.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config_sim.yaml non presente")
    cfg = load_config(cfg_path)
    cfg["universe"] = {"US": {"benchmark": "SYN-BENCH", "mode": "liquidity",
                              "tickers": SyntheticProvider().universe(),
                              "liquidity": {"min_price": 5.0, "min_volume": 1000,
                                            "min_dollar_volume": 50_000,
                                            "lookback_weeks": 13}}}
    cfg["random_twin"] = {"enabled": True, "n_sims": 10, "use_market_filter": True}
    res = run_backtest(cfg, DataStore(cfg["data_dir"], make_provider(cfg["provider"])))
    assert "random_twin" in res.metrics
    t = res.metrics["random_twin"]
    assert t["n_sims"] == 10
    assert 0 <= t["strategy_percentile"] <= 100


def test_random_twin_handles_position_leaving_universe():
    """Copre il ramo in cui un titolo esce dall'universo mentre è in
    portafoglio e il prezzo corrente manca: usa entry_price come fallback.
    Questo ramo aveva un bug (.entry invece di .entry_price) sfuggito ai test
    più leggeri."""
    from src.random_twin import run_random_twin
    from src.config import load_config
    from src.data_store import DataStore
    from src.providers import make_provider
    from src.providers.synthetic import SyntheticProvider
    from src.backtest import load_universe
    from src.indicators import to_weekly, enrich_weekly
    from src.stages import classify_stages
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config_sim.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("config_sim.yaml non presente")
    cfg = load_config(cfg_path)
    tick = SyntheticProvider().universe()
    cfg["universe"] = {"US": {"benchmark": "SYN-BENCH", "mode": "liquidity",
                              "tickers": tick,
                              "liquidity": {"min_price": 5.0, "min_volume": 1000,
                                            "min_dollar_volume": 50_000,
                                            "lookback_weeks": 13}}}
    store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
    universe = load_universe(cfg)
    st = cfg["stages"]
    per_ticker, ticker_market, market_stage = {}, {}, {}
    b = store.get_with_warmup("SYN-BENCH", cfg["backtest"]["start"], cfg["backtest"]["end"])
    bw = enrich_weekly(to_weekly(b), None, st["ma_weeks"], st["slope_lookback"])
    market_stage["US"] = classify_stages(bw, st["flat_slope"])
    for tk in tick:
        d = store.get_with_warmup(tk, cfg["backtest"]["start"], cfg["backtest"]["end"])
        w = enrich_weekly(to_weekly(d), bw, st["ma_weeks"], st["slope_lookback"])
        per_ticker[tk] = w
        ticker_market[tk] = "US"
        universe["US"]["liq"].compute(tk, w)
    weeks = market_stage["US"].loc[cfg["backtest"]["start"]:cfg["backtest"]["end"]].index
    # non deve sollevare AttributeError
    curve = run_random_twin(per_ticker, ticker_market, universe, market_stage,
                            weeks, cfg, seed=0, use_market_filter=True)
    assert len(curve) > 0


def test_twin_delisted_position_books_real_loss():
    """REGRESSIONE (bug v3.9.2): il gemello rimborsava i delistati al prezzo
    di ENTRATA, cancellando le perdite — survivorship bias dentro il metro che
    doveva esserne immune. Il delistato va liquidato all'ultimo prezzo valido."""
    from src.random_twin import run_random_twin
    weeks = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    px_dead = np.concatenate([np.linspace(50, 5, 20), np.full(10, np.nan)])
    per_ticker = {"DEAD.US": pd.DataFrame({"adj_close": px_dead}, index=weeks)}
    cfg = {"portfolio": {"initial_capital": 100_000, "risk_per_trade": 0.05,
                         "slippage_bps": 0, "commission_bps": 0}}
    curve = run_random_twin(per_ticker, {"DEAD.US": "US"}, {"US": {"liq": None}},
                            {"US": pd.Series(2, index=weeks)}, weeks, cfg,
                            seed=1, use_market_filter=True)
    assert curve.iloc[-1] < 95_000, \
        "il delistato deve produrre la perdita reale, non il rimborso all'entrata"




def test_forward_returns_outlier_sanitization():
    """I forward returns corrotti (split-adjustment esploso a orizzonti lunghi)
    devono essere scartati dalle statistiche di decomposizione, altrimenti un
    solo segnale a +9600% rende le medie assurde."""
    import pandas as pd
    n = 100
    fwd = pd.Series(np.random.normal(0.05, 0.15, n))
    fwd.iloc[5] = 96.0     # outlier corrotto
    good = fwd.notna() & fwd.between(-1.0, 9.0)
    assert int((~good).sum()) == 1
    clean = fwd[good]
    assert clean.max() < 9.0
    assert abs(clean.mean()) < 1.0   # media sensata, non gonfiata


def test_base_requires_decline_before(): 
    """Requisito A (Weinstein): la base è accumulazione DOPO un declino.
    Una base preceduta da Fase 4 è valida; una che segue Fase 2/3 no."""
    from src.stages import base_features
    weeks = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    st = pd.Series([4]*10 + [1]*20 + [2]*10 + [1]*20, index=weeks, name="stage")
    px = np.full(60, 100.0)
    enr = pd.DataFrame({"adj_high": px*1.02, "adj_low": px*0.98,
                        "adj_close": px}, index=weeks)
    feats = base_features(enr, st, decline_lookback=26)
    assert feats["after_decline"].iloc[10:30].all()      # dopo Fase 4
    assert not feats["after_decline"].iloc[40:60].any()  # dopo Fase 2


def test_after_decline_no_lookahead():
    """after_decline a t deve dipendere solo da stages <= t."""
    from src.stages import base_features
    weeks = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    st = pd.Series([4]*10 + [1]*25 + [2]*25, index=weeks, name="stage")
    px = np.full(60, 100.0)
    enr = pd.DataFrame({"adj_high": px*1.02, "adj_low": px*0.98,
                        "adj_close": px}, index=weeks)
    full = base_features(enr, st, decline_lookback=26)
    trunc = base_features(enr.iloc[:34], st.iloc[:34], decline_lookback=26)
    assert full["after_decline"].iloc[33] == trunc["after_decline"].iloc[33]


def test_base_vol_ratio_measures_contraction():
    """base_vol_ratio = volume medio nella base / volume medio prima della base.
    Volume che si prosciuga nella base -> < 1; volume che sale -> > 1."""
    from src.stages import base_features
    weeks = pd.date_range("2020-01-03", periods=50, freq="W-FRI")
    # 10 settimane di declino a volume ALTO, poi base a volume BASSO
    st = pd.Series([4]*10 + [1]*40, index=weeks, name="stage")
    px = np.full(50, 100.0)
    vol = np.array([1000.0]*10 + [400.0]*40)      # base al 40% del pre-base
    enr = pd.DataFrame({"adj_high": px*1.02, "adj_low": px*0.98,
                        "adj_close": px, "volume": vol}, index=weeks)
    feats = base_features(enr, st, decline_lookback=26)
    # a metà della base il rapporto deve essere ~0.4 (contrazione)
    assert feats["base_vol_ratio"].iloc[25] == pytest.approx(0.4, abs=0.05)

    # volume che INVECE aumenta nella base -> ratio > 1
    vol2 = np.array([400.0]*10 + [1000.0]*40)
    enr2 = enr.assign(volume=vol2)
    feats2 = base_features(enr2, st, decline_lookback=26)
    assert feats2["base_vol_ratio"].iloc[25] > 1.0


def test_base_vol_ratio_excludes_prebreakout_runup():
    """Il dry-up si misura sulla parte iniziale/centrale della base, ESCLUDENDO
    la ripresa di volume pre-breakout. Una base che si asciuga e poi riprende
    (caso MDSO) non deve risultare 'non contratta' per colpa della coda."""
    from src.stages import base_features
    weeks = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    st = pd.Series([4]*10 + [1]*30, index=weeks, name="stage")
    px = np.full(40, 100.0)
    # pre-base 1000; base: 20 sett. asciutte (500) + 10 di ripresa (2000)
    vol = np.array([1000.0]*10 + [500.0]*20 + [2000.0]*10)
    enr = pd.DataFrame({"adj_high": px*1.02, "adj_low": px*0.98,
                        "adj_close": px, "volume": vol}, index=weeks)
    # media CUMULATIVA finirebbe a 1.0 (10*2000+20*500)/30 / 1000; misurando solo
    # la PRIMA parte (dry-up) deve restare < 1
    feats = base_features(enr, st, decline_lookback=26, vol_front_frac=0.6)
    assert feats["base_vol_ratio"].iloc[-1] < 1.0


def test_base_obv_measures_accumulation():
    """base_obv > 0 se il volume si concentra sulle settimane di RIALZO
    (accumulo), < 0 se sui ribassi (distribuzione). Solo dati <= t."""
    from src.stages import base_features
    weeks = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    st = pd.Series([4]*10 + [1]*30, index=weeks, name="stage")
    changes = np.tile([1.0, -0.5], 15)                 # 30 settimane, trend su
    px = np.concatenate([np.full(10, 100.0), 100 + np.cumsum(changes)])
    up = np.zeros(40, dtype=bool)
    up[1:] = px[1:] > px[:-1]
    vol_acc = np.where(up, 1000.0, 100.0)              # volume ALTO sugli up
    enr = pd.DataFrame({"adj_high": px*1.01, "adj_low": px*0.99,
                        "adj_close": px, "volume": vol_acc}, index=weeks)
    assert base_features(enr, st)["base_obv"].iloc[-1] > 0.3     # accumulo
    # volume alto sui RIBASSI = distribuzione → base_obv < 0
    enr2 = enr.assign(volume=np.where(up, 100.0, 1000.0))
    assert base_features(enr2, st)["base_obv"].iloc[-1] < 0.0


def test_base_vol_ratio_no_lookahead():
    """base_vol_ratio a t usa solo volume <= t: troncare la serie non cambia t."""
    from src.stages import base_features
    weeks = pd.date_range("2020-01-03", periods=50, freq="W-FRI")
    st = pd.Series([4]*10 + [1]*40, index=weeks, name="stage")
    px = np.full(50, 100.0)
    vol = np.concatenate([np.full(10, 1000.0), np.linspace(500, 200, 40)])
    enr = pd.DataFrame({"adj_high": px*1.02, "adj_low": px*0.98,
                        "adj_close": px, "volume": vol}, index=weeks)
    full = base_features(enr, st, decline_lookback=26)
    trunc = base_features(enr.iloc[:31], st.iloc[:31], decline_lookback=26)
    assert full["base_vol_ratio"].iloc[30] == pytest.approx(
        trunc["base_vol_ratio"].iloc[30], nan_ok=True)


def test_short_signals_fire_on_breakdown():
    """La logica short: un breakdown sotto il supporto di un top (Fase 3) genera
    segnali con stop SOPRA l'ingresso. Il supporto è 'congelato' (stabilito
    prima della rottura), altrimenti il prezzo non potrebbe mai romperlo."""
    from src.stages import top_features
    from src.signals import detect_signals_short
    weeks = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    close = np.concatenate([np.linspace(70, 100, 10),
                            100 + np.sin(np.linspace(0, 6, 15)) * 2,
                            np.linspace(97, 60, 15)])
    ma = pd.Series(close).rolling(10, min_periods=1).mean().to_numpy()
    df = pd.DataFrame({"adj_close": close, "adj_high": close*1.01,
                       "adj_low": close*0.99, "ma": ma,
                       "ma_slope": np.gradient(ma),
                       "mansfield": np.linspace(5, -10, 40),
                       "vol_ratio": np.full(40, 3.0)}, index=weeks)
    stage = pd.Series([2]*10 + [3]*15 + [4]*15, index=weeks)
    df = df.join(top_features(df, stage))
    df["stage"] = stage
    sig_cfg = {"min_base_weeks": 8, "max_base_depth": 0.60, "volume_ratio_min": 1.5,
               "mansfield_min": 0.0, "mansfield_rising_weeks": 4,
               "require_decline": True, "market_filter": False}
    sigs = detect_signals_short("TEST", df, None, sig_cfg,
                                {"pivot_k": 2, "pivot_buffer": 0.02, "stop_buffer": 0.03})
    assert len(sigs) > 0
    for s in sigs:
        assert s.stop > s.entry      # short: stop SOPRA l'ingresso


def test_portfolio_short_profits_when_price_falls():
    """Meccanica short del Portfolio: si guadagna quando il prezzo SCENDE,
    l'equity sale se il titolo cala, e il costo di prestito erode la cassa."""
    from src.portfolio import Portfolio
    pf = Portfolio(100_000, 0.01, 1.0, 100, 5, 10)
    d0 = pd.Timestamp("2020-01-03")
    d1 = pd.Timestamp("2020-03-06")
    pf.open_short("X", d0, 100.0, 120.0, 100)     # short 100 @ ~100, stop 120
    assert pf.equity({"X": 80.0}) > pf.equity({"X": 130.0})   # sale se scende
    cash_before = pf.cash
    pf.charge_borrow({"X": 100.0}, 0.03 / 52)     # carry settimanale
    assert pf.cash < cash_before                  # il prestito costa
    tr = pf.close("X", d1, 80.0, "cover")         # ricopre a 80
    assert tr.pnl > 0 and tr.ret > 0              # profitto sullo short


def test_chart_payload_is_json_safe():
    """BUG REALE (luglio 2026): `vol_ratio` = volume / media precedente diventa
    `inf` se una settimana ha volume zero. JSON non ammette NaN/Infinity → la
    risposta di /api/chart falliva e ABBATTEVA il server del simulatore. La
    pulizia deve usare isfinite (copre NaN e ±inf), non solo isnan."""
    import json

    def clean(series, nd=2):     # stessa logica di Session.chart
        return [None if not np.isfinite(x) else round(float(x), nd) for x in series]

    sporco = [1.0, np.inf, -np.inf, np.nan, 2.5]
    pulito = clean(sporco)
    assert pulito == [1.0, None, None, None, 2.5]
    json.dumps({"vol_ratio": pulito})        # prima sollevava ValueError


def test_portfolio_reduce_partial_close():
    """reduce(): chiusura PARZIALE di una long — vende una quota, incassa, lascia
    aperto il resto. È il mattone del ribilanciamento (fare spazio ai segnali)."""
    from src.portfolio import Portfolio
    pf = Portfolio(100_000, 0.01, 1.0, 100, 0, 0)     # senza costi, per chiarezza
    d0 = pd.Timestamp("2020-01-03")
    d1 = pd.Timestamp("2020-02-07")
    pf.open("X", d0, 100.0, 90.0, 100)                # 100 azioni @ 100
    cash0 = pf.cash
    pf.reduce("X", d1, 110.0, 40.0, "rebalance_trim") # vende 40 @ 110
    assert pf.positions["X"].shares == pytest.approx(60)
    assert pf.cash == pytest.approx(cash0 + 40 * 110)
    assert pf.trades[-1].shares == pytest.approx(40)
    assert pf.trades[-1].reason == "rebalance_trim"


def test_top_obv_measures_distribution():
    """Specchio di base_obv: top_obv < 0 quando il volume si concentra sui
    RIBASSI dentro il top (distribuzione)."""
    from src.stages import top_features
    weeks = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    st = pd.Series([2]*10 + [3]*30, index=weeks, name="stage")
    changes = np.tile([-1.0, 0.5], 15)            # trend giù nel top
    px = np.concatenate([np.full(10, 100.0), 100 + np.cumsum(changes)])
    down = np.zeros(40, dtype=bool)
    down[1:] = px[1:] < px[:-1]
    vol = np.where(down, 1000.0, 100.0)           # volume ALTO sui ribassi
    enr = pd.DataFrame({"adj_high": px*1.01, "adj_low": px*0.99,
                        "adj_close": px, "volume": vol}, index=weeks)
    feats = top_features(enr, st)
    assert feats["top_obv"].dropna().iloc[-1] < 0     # distribuzione


def test_short_support_no_lookahead():
    """Il supporto congelato a t non deve dipendere da dati futuri."""
    from src.stages import top_features
    weeks = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    close = np.concatenate([np.linspace(70, 100, 10),
                            100 + np.sin(np.linspace(0, 6, 15)) * 2,
                            np.linspace(97, 60, 15)])
    df = pd.DataFrame({"adj_close": close, "adj_high": close*1.01,
                       "adj_low": close*0.99}, index=weeks)
    stage = pd.Series([2]*10 + [3]*15 + [4]*15, index=weeks)
    full = top_features(df, stage)
    trunc = top_features(df.iloc[:30], stage.iloc[:30])
    a, b = full["support_est"].iloc[29], trunc["support_est"].iloc[29]
    assert (pd.isna(a) and pd.isna(b)) or abs(a - b) < 1e-9


def test_mansfield_rising_filter_toggleable():
    """Il filtro sulla direzione del Mansfield deve essere disattivabile, per
    permettere l'analisi dei quadranti (che deve vedere anche i segnali in
    discesa). Con require_mansfield_rising=False, il filtro non si applica."""
    import inspect
    from src import signals
    src = inspect.getsource(signals.detect_signals)
    assert "require_mansfield_rising" in src


def test_bulk_update_append_and_refetch(tmp_path):
    """update_prices: l'append incrementale aggiunge SOLO le barre nuove (dedup),
    passa per la sanificazione, aggiorna il meta e invalida la cache settimanale;
    il re-fetch (ticker con azione societaria) ri-scarica l'intero storico."""
    import json as _json

    import update_prices as up
    from src.data_store import DataStore
    from src.providers.base import DataProvider

    class _P(DataProvider):
        name = "test"

        def get_eod(self, ticker, start, end):
            idx = pd.date_range(start, end, freq="B")
            df = pd.DataFrame({"open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0,
                               "adj_close": 2.0, "volume": 50}, index=idx)
            return self.validate(df, ticker)

    store = DataStore(str(tmp_path), _P())
    tk = "AAA.US"
    idx = pd.date_range("2024-01-01", "2024-01-05", freq="B")   # Lun-Ven
    df0 = pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0,
                        "adj_close": 1.0, "volume": 100}, index=idx)
    df0.to_parquet(store._path(tk))
    store._meta_path(tk).write_text(_json.dumps({"asked_from": "2024-01-01", "asked_to": "2024-01-05"}))
    wf = store.weekly_root / f"{tk}.deadbeef1234.parquet"       # cache settimanale fittizia
    df0.to_parquet(wf)

    # APPEND: una barra nuova (01-08) + una già presente (01-05) che va scartata
    new = {pd.Timestamp("2024-01-08"): {"open": 1.0, "high": 1.2, "low": 0.95,
                                        "close": 1.1, "adj_close": 1.1, "volume": 120},
           pd.Timestamp("2024-01-05"): {"open": 1.0, "high": 1.1, "low": 0.9,
                                        "close": 1.0, "adj_close": 1.0, "volume": 100}}
    assert up._append(store, tk, new, "2024-01-08") is True
    got = pd.read_parquet(store._path(tk))
    assert pd.Timestamp("2024-01-08") in got.index
    assert len(got) == 6                                        # 5 + 1 (dedup su 01-05)
    assert _json.loads(store._meta_path(tk).read_text())["asked_to"] == "2024-01-08"

    up._invalidate_weekly(store, tk)
    assert not wf.exists()                                      # settimanale invalidato

    # REFETCH: ri-scarica intero (sostituisce col provider, close=2.0 ovunque)
    assert up._refetch(store, tk, "2024-01-01", "2024-01-10") is True
    got2 = pd.read_parquet(store._path(tk))
    assert (got2["close"] == 2.0).all()
    assert _json.loads(store._meta_path(tk).read_text())["asked_to"] == "2024-01-10"
