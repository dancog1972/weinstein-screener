import json, pathlib
f = json.load(open("data/failed_tickers.json"))
print(f"falliti: {len(f)}")
for x in f[:10]:
    print(f'  {x["ticker"]:22s} {x["error"][:95]}')
print()
tipi = {}
for x in f:
    k = x["error"].split(":")[0][:60]
    tipi[k] = tipi.get(k, 0) + 1
print("errori raggruppati:")
for k, v in sorted(tipi.items(), key=lambda i: -i[1]):
    print(f"  {v:4d} x  {k}")
