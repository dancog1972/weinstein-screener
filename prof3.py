import sys, time, pathlib
sys.path.insert(0, ".")
from src.data_store import DataStore
from src.providers import make_provider
from src.config import load_config
from src.indicators import to_weekly, enrich_weekly
from src.backtest import prepare_ticker

cfg = load_config("config_us_liquidity.yaml")
store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
st = cfg["stages"]; start = cfg["backtest"]["start"]; end = cfg["backtest"]["end"]

bench_d = store.get_with_warmup(cfg["universe"]["US"]["benchmark"], start, end)
bench_w = enrich_weekly(to_weekly(bench_d), None, st["ma_weeks"], st["slope_lookback"])
wkey = store.weekly_key(cfg["universe"]["US"]["benchmark"], st["ma_weeks"],
                        st["slope_lookback"], st["flat_slope"], start, end)
print("weekly_key di questo run:", wkey)

# quante chiavi diverse ci sono in cache? se piu di una, si rigenera ogni volta
keys = set()
for f in store.weekly_root.glob("*.parquet"):
    parts = f.stem.split(".")
    if len(parts) >= 2: keys.add(parts[-1])
print("chiavi distinte in cache:", len(keys))
for k in list(keys)[:5]: print("  ", k)

# il file per questa chiave esiste gia?
import pandas as pd
tk = cfg["universe"]["US"]["benchmark"]
wp = store._weekly_path("AAPL.US", wkey)
print("cache per AAPL con questa chiave esiste:", wp.exists())
