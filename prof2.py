import time, sys, pandas as pd
sys.path.insert(0, ".")
from src.data_store import DataStore
from src.providers import make_provider
from src.config import load_config
from src.indicators import confirmed_pivot_lows
from src.universe import LiquidityUniverse

cfg = load_config("config_us_liquidity.yaml")
store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))
wr = store.weekly_root
files = list(wr.glob("*.parquet"))[:300]

dfs = [pd.read_parquet(f) for f in files]
lq = cfg["universe"]["US"]["liquidity"]

t = time.perf_counter()
for df in dfs: confirmed_pivot_lows(df, 2)
piv_ms = (time.perf_counter()-t)/len(dfs)*1000

liq = LiquidityUniverse(min_price=lq["min_price"], min_volume=lq["min_volume"],
    min_dollar_volume=lq["min_dollar_volume"], lookback_weeks=lq["lookback_weeks"])
t = time.perf_counter()
for i, df in enumerate(dfs): liq.compute(f"T{i}", df)
liq_ms = (time.perf_counter()-t)/len(dfs)*1000

print(f"confirmed_pivot_lows: {piv_ms:.2f} ms/titolo -> {piv_ms*14564/1000/60:.1f} min")
print(f"LiquidityUniverse.compute: {liq_ms:.2f} ms/titolo -> {liq_ms*14564/1000/60:.1f} min")
