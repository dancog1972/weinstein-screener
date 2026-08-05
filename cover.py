import pandas as pd, pathlib, random
files = [f for f in pathlib.Path("data/eodhd").glob("*.parquet")]
sample = random.sample(files, min(30, len(files)))
ends = []
for f in sample:
    try:
        df = pd.read_parquet(f)
        if len(df): ends.append(df.index.max())
    except Exception: pass
s = pd.Series(ends)
print(f"campione {len(ends)} titoli - data finale in cache:")
print(f"  mediana: {s.median().date()} - piu recente: {s.max().date()}")
n = (s < pd.Timestamp("2024-12-21")).sum()
print(f"NON arrivano a fine 2024: {n}/{len(ends)}")
