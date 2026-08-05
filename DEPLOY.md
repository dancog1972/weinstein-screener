# Deploy dello screener su GitHub (Pages + Actions)

Il job gira ogni **sabato** (o a mano): aggiorna i prezzi via bulk, genera lo
screener e pubblica la **pagina interattiva** su GitHub Pages. Cache dati come
**Release asset** (non in git). Costo **€0** (repo pubblico → minuti Actions
illimitati).

## Setup una tantum

### 1. Rigenera la chiave EODHD
La vecchia è comparsa in chiaro durante lo sviluppo: rigenerala dal pannello EODHD
e usa solo la nuova.

### 2. Crea il repo pubblico e pusha
```bash
git init && git add . && git commit -m "screener"
gh repo create weinstein-screener --public --source=. --push
```
(`data/` e `output/` sono in `.gitignore`; `assets/plotly.min.js` va committato.)

### 3. Aggiungi il Secret con la chiave
Repo → Settings → Secrets and variables → Actions → New repository secret:
- **Name**: `EODHD_API_KEY`  ·  **Value**: la chiave NUOVA

### 4. Abilita Pages
Repo → Settings → Pages → **Source: GitHub Actions**.

### 5. Carica la cache iniziale (dal tuo PC, con la cache piena)
```bash
python pack_cache.py --out data_cache.tar.gz          # ~300-500 MB, attivi + 6 anni
gh release create data-cache data_cache.tar.gz -t "cache dati" -n "cache screener"
```

### 6. Primo run
Repo → Actions → **Screener settimanale** → *Run workflow*.
Al termine la pagina è su `https://<utente>.github.io/weinstein-screener/`
(interattiva = `index.html`, report statico = `report.html`).

## Come funziona il job
1. **Ripristina** la cache dal Release asset `data-cache`.
2. **`update_prices.py`** — bulk incrementale (append + refetch dei ticker con
   split/dividendo). Usa il Secret.
3. **`screener.py`** — `screener.json`/`.html` (report validato) +
   `screener_app_data.json` (superset). Usa il Secret (aggiorna benchmark/ETF).
4. **`build_page.py`** — `screener_app.html` (pagina autonoma, offline dalla cache).
5. **Pubblica** su Pages e **risalva** la cache aggiornata nel Release asset.

## Note
- Cambi ai parametri del segnale → committi il config, il prossimo run li usa.
- La cache settimanale (`_weekly`) NON viaggia: si ricalcola a ogni run.
- Se salti settimane, `update_prices` recupera il gap (fino a `--max-gap-days`,
  default 45; oltre, conviene un full refresh locale + nuovo `pack_cache`).
- Portafoglio fittizio = fase 2 (vedi discussione: Modello 2+, DB gratuito).
