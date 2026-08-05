"""Caricamento e validazione della configurazione (config.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP = ["provider", "universe", "backtest", "stages", "signal", "exits", "portfolio"]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    missing = [k for k in REQUIRED_TOP if k not in cfg]
    if missing:
        raise ValueError(f"Config incompleta, mancano le sezioni: {missing}")
    cfg.setdefault("data_dir", "data")
    cfg.setdefault("output_dir", "output")
    cfg["_config_path"] = str(path)
    cfg["_raw_yaml"] = path.read_text(encoding="utf-8")
    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    st, sig, pf = cfg["stages"], cfg["signal"], cfg["portfolio"]
    assert st["ma_weeks"] > 4, "ma_weeks troppo piccolo"
    assert 0 < sig["volume_ratio_min"], "volume_ratio_min deve essere > 0"
    assert sig["min_base_weeks"] >= 2, "min_base_weeks deve essere >= 2"
    assert 0 < pf["risk_per_trade"] < 0.2, "risk_per_trade fuori range (0, 0.2)"
    assert 0 < pf["max_position_pct"] <= 1, "max_position_pct fuori range (0, 1]"
    bt = cfg["backtest"]
    assert bt["start"] < bt["end"], "backtest.start deve precedere backtest.end"
    assert bt["start"] <= bt["oos_split"] <= bt["end"], "oos_split fuori periodo"
