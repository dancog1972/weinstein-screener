# Backend watchlist — Cloudflare Workers + D1 (gratis)

Abilita i bottoni **★ segui** sui candidati "quasi": clicchi, la scelta va nel DB
D1 tramite un Worker, e il job settimanale la mostra nel panel "QUASI seguiti"
della pagina Follow-up. Gratis, sempre acceso, nessuna carta di credito.

**Sicurezza:** ogni scrittura richiede un **segreto** che scegli tu. Non è nella
pagina: lo digiti UNA volta nel browser (resta in localStorage). Solo tu puoi
modificare la watchlist.

## Prerequisiti
- Un **account Cloudflare** gratis: [dash.cloudflare.com/sign-up](https://dash.cloudflare.com/sign-up)
- **Node.js** (serve a `wrangler`). Se manca: `winget install OpenJS.NodeJS.LTS`, poi riapri il terminale.

## Setup (una volta), dalla cartella `cloudflare/`
```bash
cd C:\users\daniele\botsfinanza\weinstein-botv2\cloudflare
```

**1. Installa e collega wrangler** (la CLI di Cloudflare):
```bash
npm install -g wrangler
```
```bash
wrangler login
```
(si apre il browser, autorizza)

**2. Crea il database D1:**
```bash
wrangler d1 create weinstein-watch
```
Stampa un `database_id`. **Copialo** e incollalo in `wrangler.toml` al posto di
`INCOLLA_QUI_DOPO_wrangler_d1_create`.

**3. Crea la tabella:**
```bash
wrangler d1 execute weinstein-watch --remote --file=schema.sql
```

**4. Imposta il segreto** (scegline uno tuo, ricordalo — lo userai nel browser):
```bash
wrangler secret put WATCH_SECRET
```
(incolla il segreto quando lo chiede)

**5. Pubblica il Worker:**
```bash
wrangler deploy
```
Stampa l'URL, tipo `https://weinstein-watch.<tuo-subdominio>.workers.dev`. **Copialo.**

## Collega al sito (GitHub Secrets)
Nel repo `weinstein-screener` → Settings → Secrets and variables → Actions, aggiungi:
- **`WATCH_URL`** = l'URL del Worker (punto 5)
- **`WATCH_SECRET`** = lo stesso segreto del punto 4

Poi **Actions → Run workflow**: la pagina avrà i bottoni **★ segui**, e la pagina
Follow-up il panel "QUASI seguiti".

## Uso
Nello screener, apri un candidato (scheda col grafico) → **★ segui**. La prima volta
ti chiede il **segreto** (quello del punto 4): lo digiti una volta, resta nel browser.
Da lì i titoli seguiti compaiono nel panel inferiore della pagina Follow-up,
tracciati come i pieni (% da allora, stop colpito). Ri-clicca **✓ seguito** per togliere.

## Note
- Il Worker è ~50 righe (`worker.js`), il DB è D1 (SQLite). Tetto gratis: 100k
  richieste/giorno, 500 MB — abbondante.
- Questo backend servirà **anche al portafoglio** (fase 2): stessa infrastruttura.
- Se non configuri nulla, tutto funziona lo stesso **senza** i bottoni ★ (degrada
  ai soli pieni automatici).
