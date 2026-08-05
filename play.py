"""Server FastAPI del simulatore human-vs-bot.

Avvio:
  python play.py                       # usa config_sim.yaml, apre su :8000
  python play.py --config altro.yaml --port 8080

Poi apri http://localhost:8000 nel browser.

Endpoint:
  GET  /                  -> la UI (pagina singola)
  POST /api/new           -> crea sessione {blind: bool} -> stato iniziale
  POST /api/advance       -> avanza al prossimo evento -> stato
  POST /api/decide        -> {alias, action:'buy'|'skip', shares?} -> esito
  GET  /api/chart/{alias} -> serie del titolo tagliata al presente
  GET  /api/curves        -> curve equity human/bot/benchmark (a fine partita)
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
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from src.config import load_config
from src.simulator import Session

app = FastAPI(title="Weinstein Simulator")
_STATE: dict[str, object] = {"cfg": None, "session": None}


class NewReq(BaseModel):
    blind: bool = True


class DecideReq(BaseModel):
    alias: str
    action: str                       # 'buy' | 'limit' | 'skip'
    shares: float | None = None
    limit_price: float | None = None
    stop_mode: str = "bot"            # 'bot' | 'pivot' | 'manual'
    stop_price: float | None = None


def _session() -> Session:
    s = _STATE.get("session")
    if s is None:
        raise HTTPException(400, "nessuna sessione attiva: chiama /api/new")
    return s  # type: ignore[return-value]


@app.post("/api/new")
def api_new(req: NewReq):
    cfg = _STATE["cfg"]
    s = Session(cfg, blind=req.blind)  # type: ignore[arg-type]
    _STATE["session"] = s
    return s.advance()


@app.post("/api/advance")
def api_advance():
    return _session().advance()


@app.post("/api/decide")
def api_decide(req: DecideReq):
    return _session().decide(req.alias, req.action, req.shares, req.limit_price,
                             req.stop_mode, req.stop_price)


@app.get("/api/chart/{alias}")
def api_chart(alias: str):
    return _session().chart(alias)


@app.get("/api/curves")
def api_curves():
    return _session().equity_curves()


@app.get("/api/export")
def api_export():
    """Partita completa in JSON: journal delle decisioni, operazioni,
    divergenze human-vs-bot, curve. È il file da conservare o condividere."""
    import json
    from fastapi.responses import JSONResponse
    s = _session()
    data = s.export()
    fn = f"partita_{s.id}.json"
    return JSONResponse(data, headers={"Content-Disposition": f'attachment; filename="{fn}"'})


@app.get("/api/export.csv")
def api_export_csv():
    """Solo le operazioni chiuse, in CSV per Excel."""
    s = _session()
    fn = f"operazioni_{s.id}.csv"
    return Response(s.export_trades_csv(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{fn}"'})


@app.get("/plotly.js")
def plotly_js():
    """Serve Plotly dal pacchetto Python installato localmente, così il
    grafico funziona anche senza accesso alla CDN esterna (offline-safe)."""
    import plotly
    from pathlib import Path as _P
    js = _P(plotly.__file__).parent / "package_data" / "plotly.min.js"
    if not js.exists():
        raise HTTPException(500, "bundle plotly non trovato nel pacchetto installato")
    return Response(js.read_text(encoding="utf-8"), media_type="application/javascript")


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "src" / "sim_ui.html").read_text(encoding="utf-8")


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_sim.yaml")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    _STATE["cfg"] = load_config(args.config)
    print(f"\n  Simulatore pronto → http://localhost:{args.port}\n"
          f"  (Ctrl+C per fermare)\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
