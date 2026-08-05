import pandas as pd
df = pd.read_parquet("data/universe_US.parquet")
print(f"totale: {len(df)}\n")
print("colonne:", list(df.columns), "\n")
if "type" in df.columns:
    print("TYPE:"); print(df["type"].value_counts().to_string(), "\n")
if "exchange_real" in df.columns:
    print("BORSA REALE:"); print(df["exchange_real"].value_counts().head(12).to_string(), "\n")
if "currency" in df.columns:
    print("VALUTA:"); print(df["currency"].value_counts().head(5).to_string(), "\n")
print("delistati:", int(df["is_delisted"].sum()))
print("\nprimi 15 ticker:", ", ".join(df["ticker"].head(15)))
