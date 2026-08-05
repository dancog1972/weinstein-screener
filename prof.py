import time, pathlib, pandas as pd, sys
sys.path.insert(0, ".")
from src.data_store import DataStore
from src.providers import make_provider
from src.config import load_config

cfg = load_config("config_us_liquidity.yaml")
store = DataStore(cfg["data_dir"], make_provider(cfg["provider"]))

# quanti file settimanali in cache? (la prova che la cache è popolata)
wr = store.weekly_root
n_weekly = len(list(wr.glob("*.parquet"))) if wr.exists() else 0
print(f"file settimanali in cache: {n_weekly}")

# tempo di lettura di 200 settimanali cachati
files = list(wr.glob("*.parquet"))[:200]
t = time.perf_counter()
for f in files: pd.read_parquet(f)
if files:
    ms = (time.perf_counter()-t)/len(files)*1000
    print(f"lettura settimanale cachato: {ms:.2f} ms/titolo")
    print(f"stima su 14564: {ms*14564/1000/60:.1f} min di sola lettura cache")
