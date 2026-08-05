import json, pathlib
p = sorted(pathlib.Path("data/eodhd").glob("*.parquet"))
print(f"ticker in cache: {len(p)}")
f = pathlib.Path("data/failed_tickers.json")
if f.exists():
    fail = json.load(open(f))
    print(f"falliti: {len(fail)}  ({len(fail)/(len(p)+len(fail))*100:.0f}%)")
    tipi = {}
    for x in fail:
        k = x["error"].split(":")[0][:50]
        tipi[k] = tipi.get(k, 0) + 1
    for k, v in sorted(tipi.items(), key=lambda i: -i[1])[:4]:
        print(f"  {v:4d} x {k}")
else:
    print("nessun fallito")
etf = [x.stem for x in p if x.stem.split(".")[0] in
       ("XLK","XLF","XLV","XLY","XLP","XLI","XLE","XLB","XLU","XLRE","XLC")]
print(f"\nETF settoriali: {len(etf)}/11  {sorted(etf)}")
print(f"benchmark SPY: {'SI' if any(x.stem.startswith('SPY') for x in p) else 'NO'}")
