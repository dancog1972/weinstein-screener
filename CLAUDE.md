# Weinstein Bot — contesto del progetto

## Chi sono e come lavorare con me

Daniele, in Francia. Parlami **in italiano**. Preferisco risposte **concise**.

**Regola importante**: prima di eseguire modifiche non banali, **presentami le opzioni
con costi e rischi** e lasciami decidere. Non partire a codificare di tua iniziativa.
Questa regola nasce da un episodio reale: mi era stata promessa un'ottimizzazione
risolutiva che aveva ottimizzato il pezzo sbagliato, senza aver prima profilato.

Leggo Weinstein da vicino e correggo la metodologia quando serve. Se sbaglio un
concetto, dimmelo.

## Cos'è questo progetto

Backtest del metodo a 4 fasi di Stan Weinstein sull'intero universo azionario USA
(~20.900 titoli, 2001–2024), costruito con ossessione per l'eliminazione dei bias.

**Ambiente**: Windows PowerShell, Python 3.14, path `C:\users\daniele\botsfinanza\weinstein-botv2`

## L'OBIETTIVO VERO (non fraintenderlo)

La domanda non è "batto l'S&P?" (market timing). È:

> **La selezione dei titoli secondo Weinstein aggiunge valore rispetto a scegliere
> a caso dallo stesso universo liquido, con capitale finito?**

Il metro di paragone giusto è il **gemello casuale** (`src/random_twin.py`), non il
benchmark. Cento portafogli identici per capitale/costi/vincoli che pescano a caso.

## LE CONCLUSIONI RAGGIUNTE (con i numeri)

Il lavoro di validazione è **concluso**. Quattro ipotesi testate, tutte smentite:

| Domanda | Risposta | Numeri |
|---|---|---|
| La strategia batte il mercato? | No | CAGR +0.9% vs +8.8% SPY; MaxDD −42% vs −55%; Sharpe 0.13 vs 0.57 |
| La selezione batte il caso? | **Sì, di poco** (rivisto) | 0° col metro VECCHIO (confondente); **59° percentile** col metro equo (pari esposizione+stop). Vedi ⚠️ sotto |
| Alzare il rischio aiuta? | No | 1%/2%/5% → CAGR +0.9/+0.8/+2.4%, MaxDD peggiora. Leva su edge negativo = solo perdite |
| Gli ingressi sono buoni (stop cattivi)? | No | Solo 44-47% dei segnali batte il mercato a 4/13/26/52w = **monetina** |
| Una base più fedele (A+D) aiuta? | No | Da 1238 a 406 segnali, qualità invariata |
| Una caratteristica seleziona i vincitori? | No | Mansfield/base_len/depth/volume: nessuna separazione coerente, solo rumore |
| Lo short in Fase 4 funziona? | No | Solo 32 segnali in 24 anni, 43% batte il mercato, rendimenti short negativi |
| **Il filtro Mansfield aiuta?** | **SÌ** (unico) | Mansfield ≥0: 46% batte il mercato; <0: 32%. **È l'unico filtro con effetto reale** |

> ⚠️ **Questa riga è SUPERATA e il suo verdetto RIBALTATO.** Lo 0° percentile era un metro
> CONFONDENTE: il gemello era investito ~92% e senza stop, la strategia ~28% con stop. Con
> il gemello reso equo (**pari esposizione + stessi stop**) la strategia è al **59°
> percentile — SOPRA la mediana**: la selezione di Weinstein aggiunge un valore modesto ma
> POSITIVO. Vedi "CORREZIONE E VERIFICA" → correzione del metro.

**Conclusione (aggiornata luglio 2026 — vedi "CORREZIONE DEL METRO"):** il metodo NON
batte il mercato in assoluto (+4.5% vs +8.8% SPY: resta troppo in cassa, è market timing).
Ma **la SELEZIONE**, misurata con un metro equo (gemello a pari esposizione e stessi stop),
è al **59° percentile — batte di poco il caso**: aggiunge un valore modesto ma positivo.
La vecchia affermazione "non genera alpha / 0° percentile" era viziata da un metro
confondente (esposizione + uscite). Restano validi: il metodo non supera il benchmark
(essere long in un toro ce l'hanno tutti), e il valore vero è amplificato dalla
**discrezionalità** (pochi titoli scelti col giudizio, non 20.000 meccanicamente) →
direzione scanner + giudizio umano.

## DECISIONI DA NON RIBALTARE

- **Mai ottimizzare i parametri per far salire il CAGR.** È overfitting. I parametri
  sono scelti su ragioni metodologiche (cosa dice Weinstein), non sui risultati.
- **`volume_ratio_min: 2.0` non si abbassa.** È l'88° percentile della distribuzione
  empirica reale dei breakout. Abbassarlo per avere più segnali = overfitting.
- **Non alzare il rischio prima di avere un edge positivo.** Dimostrato coi dati.
- **`max_volatility: null`** (disattivato): l'8% era tarato su dati sintetici con ATR%
  irrealistico. Sui dati veri la mediana ATR% è ~17%, la soglia tagliava l'85% delle
  settimane eleggibili.
- Il risultato OOS che conferma l'IS **è il valore**, non un problema da aggiustare.

## BIAS ELIMINATI (non reintrodurli)

- **Survivorship**: universo completo con 14.138 delistati. Un titolo morto nel 2008
  esiste fino al 2008 e poi sparisce (`last_bar` in `backtest.py`). **Mai** usare
  `asof()` per propagare prezzi oltre la morte.
- **Look-ahead**: ogni indicatore usa solo dati ≤ t. Il segnale confronta con
  `resistance.shift(1)`. Test di troncamento a presidio.
- **Esecuzione**: `execution: next_open` — si compra all'apertura della settimana
  DOPO il segnale, non al prezzo che ha rivelato il segnale.
- **Stop sul weekly close**: `stop_intraweek: false` (dal luglio 2026) — valutato sulla
  CHIUSURA settimanale, non sul minimo intraweek. Un metodo settimanale non deve uscire
  su un wick corrotto rientrato entro venerdì (caso MDSO). `true` resta disponibile ma
  espone ai low sporchi.
- **Dati sporchi**: `providers/base.py` sanifica prezzi ≤0, OHLC incoerenti, fattori
  split corrotti, **e clippa gli spike intraday >40% dal corpo** (low/high corrotti che
  rientrano: avvelenavano minimo settimanale, pivot, base, ATR e causavano uscite false).
- **Gemello casuale**: i delistati si liquidano all'ultimo prezzo valido, NON al prezzo
  di entrata (era un bug: cancellava le perdite = survivorship nel metro stesso).

## BIAS RESIDUI DICHIARATI (nel report HTML)

1. Mappa ticker→settore **statica** (classificazione odierna applicata al passato)
2. Delisting chiuso all'ultimo prezzo noto: onesto per fusioni, **ottimistico** per
   fallimenti (il piano dati non dà il motivo del delisting). Toggle `delisting_value`:
   testato a `zero` (bound pessimista) → strategia −0.6% CAGR / −31% DD ma **1° percentile**,
   la conclusione regge. Default `last_price`.
3. Esecuzione a `next_open` — è già la scelta onesta, dichiarata per trasparenza

## ARCHITETTURA

**Dati**: EODHD, formato ticker **`.US`** (non `.NYSE`/`.NASDAQ` — bug storico).
Cache parquet in `data/eodhd/`. **La cartella `data/` va preservata**: contiene ~20.900
parquet scaricati, e `data/universe_US.parquet` si rigenera con
`python build_universe.py screen --exchange US`.

**Moduli chiave** (`src/`):
- `console.py` — riconfigura stdout in UTF-8. **Necessario**: la console Windows è
  cp1252 e crasha su `→ × ✓`. Importato da tutti gli entry point.
- `data_store.py` — cache daily + weekly-enriched (speedup 13.7×). `get_with_warmup`
  controlla `asked_to` nel meta: senza, i delistati richiamavano l'API a ogni run
  (3 ore → 15 min). **`WEEKLY_SCHEMA_VERSION`** è nella `weekly_key`: bumpalo quando
  cambi la logica del settimanale, o la cache stale viene riusata in silenzio.
- `indicators.py` — `to_weekly` (W-FRI, 62% del tempo di calcolo), `enrich_weekly`
  (aggiunge adj_high/low/open), `mansfield_rs`, `volume_ratio`, pivot confermati.
- `stages.py` — `classify_stages` (4 fasi), `base_features` (basi + `after_decline` +
  `base_vol_ratio` = contrazione volume nella base), `top_features` (top + `support_est`
  **congelato**: senza, il prezzo non potrebbe mai rompere il proprio minimo).
- `signals.py` — `detect_signals` (long), `detect_signals_short` (breakdown).
- `portfolio.py` — sizing a rischio fisso, costi 15bps/lato.
- `backtest.py` — `_prepare_backtest` (fase pesante) + simulazione portafoglio.
- `random_twin.py` — gemello vettorizzato (pannello precalcolato: 100 sorteggi in
  ~4 min invece di 5 ore).

**Entry point**:
- `run.py backtest` — backtest completo + report HTML
- `analyze.py --measure entries|characteristics|entries-short|mansfield` — decomposizione
- `build_universe.py screen --exchange US` — rigenera l'universo
- `diagnose.py` — funnel diagnostico

**Test**: 103 test in `tests/test_engine.py`. Presidiano anti-lookahead,
anti-survivorship, correttezza calcoli (inclusi contrazione volume e clip spike).
**Lanciali sempre dopo una modifica**: `python -m pytest tests/ -q`. NB: i test usano
`tmp_path` per la data_dir — non devono MAI scrivere nella `data/` reale (bug storico:
cancellavano `universe_US.parquet`).

## PARAMETRI ATTIVI (`config_us_liquidity.yaml`)

```
Universo: min_price 5, min_volume 200k, min_dollar_volume 5M, lookback 13w (no OTC)
Fasi: MA30, slope su 4 settimane, flat_slope 0.005
Segnale: base ≥12w, profondità ≤30%, require_decline true, volume ≥2.0×,
         base_volume_max_ratio 1.0 (contrazione volume nella base),
         max_breakout_stretch 0.25 (scarta gli spike da notizia: close >25% sopra resistenza),
         mansfield ≥0 e in salita da 4w, market_filter + sector_filter true
Esecuzione: next_open · Stop: pivot strutturale, buffer 2%, sulla CHIUSURA (stop_intraweek false)
Delisting: last_price (toggle delisting_value, testato anche a zero)
Portafoglio: 100k€, rischio 1%, max_position_pct 0.20, nessun tetto sul numero
```

> Nota: `max_position_pct: 0.20` nel config attuale (tetto del 20% del capitale per
> posizione): rete di sicurezza sopra lo stop pivot, evita che uno stop stretto
> (pivot vicino all'entrata) concentri troppo capitale su un solo titolo.

**Nota sul rischio**: `risk_per_trade: 0.01` è la **perdita se scatta lo stop**, NON
la quota investita. Con entry 100 e stop 90, 1% di 100k = 1000€ di rischio ÷ 10€ per
azione = 100 azioni = **10.000€ di posizione** (10% del capitale). Per questo serve
`max_position_pct`: uno stop stretto darebbe posizioni enormi.

## STATO E POSSIBILI DIREZIONI

Il progetto ha risposto alla sua domanda. Tre strade aperte, nessuna è "aggiustare
i parametri":

1. **Scanner + giudizio umano** — il sistema propone i candidati della settimana, la
   selezione finale è mia. È come operava Weinstein davvero.
2. **Fondamentali via SEC EDGAR** (gratuito, con **date di deposito reali** = vero
   point-in-time). Testare se "segnale + fondamentali sani" batte "segnale da solo".
   Cautela: i fondamentali sono il terreno dove il look-ahead rientra più facilmente
   (i bilanci si pubblicano con mesi di ritardo e vengono revisionati).
3. **Accettare il risultato** e considerare chiuso il capitolo.

## CORREZIONE E VERIFICA — luglio 2026

La richiesta dei **grafici visivi** (fasi/basi/segnali su titoli reali) è **completata**:
`make_charts.py` produce PNG dei 5 segnali migliori e 5 peggiori per rendimento a 52w,
con fasi colorate, basi+resistenza, entry/stop, breakout scartati col motivo, e l'**uscita
reale** del trade. Nel farlo sono emersi e sono stati corretti BUG REALI che inquinavano
i risultati precedenti (i vecchi numeri venivano da una pipeline difettosa):

- **Cache settimanale STALE (silenziosa).** La `weekly_key` non versionava il codice: una
  cache scritta prima di `after_decline`/`adj_open` veniva riusata, **disattivando in
  silenzio `require_decline`** e facendo ripiegare `next_open` su `adj_close` (look-ahead
  reintrodotto). Fix: `DataStore.WEEKLY_SCHEMA_VERSION` nella key (**bumpalo** se cambi
  enrich_weekly/stages/base_features) + `detect_signals` ALZA se manca `after_decline`.
- **Leak OTC.** `build_universe` non escludeva `OTCCE`/`OTCBB` → 1.858 titoli OTC (pink-sheet,
  sub-penny) nell'universo, che dominavano i "migliori/peggiori". Fix: escluso ogni
  `exchange_real` che inizia per `OTC`. Universo 20.900 → **19.042**.
- **Sub-penny nei segnali.** `signal_stats` includeva segnali non idonei per liquidità
  (il filtro era applicato solo nel portafoglio). Fix: `_prepare_backtest` filtra i segnali
  per `is_eligible` → l'analisi riflette solo i tradeabili.
- **Uscite false da dati sporchi.** Un `low` corrotto (MDSO 2012-10-31: low 31.30 su barra
  chiusa a 42.02, primo giorno post-uragano Sandy) faceva scattare lo stop intraweek e
  buttava fuori un vincitore (+29% reale invece di +122%). Fix: `stop_intraweek: false`
  (stop sulla chiusura, fedele al metodo weekly) + clip degli spike >40% in `providers/base.py`.

**Risultati sulla pipeline CORRETTA** (universo 19.042, no OTC/penny, next_open e
require_decline realmente attivi):

| configurazione | CAGR | MaxDD | percentile vs gemello |
|---|---|---|---|
| baseline pulito | +1.8% | −8.0% | 0° |
| + contrazione volume (`base_vol_ratio` ≤1) | +1.7% | −6.4% | 0° |
| + stop su chiusura + clip dati | +2.0% | −5.3% | 0° |
| + delisting → 0 (bound pessimista) | −0.6% | −31.3% | 1° |

**⚠️ CORREZIONE DEL METRO (esposizione) — la scoperta più importante.** Il gemello
classico è INVESTITO ~92% del tempo (riempie la cassa di titoli a caso e li tiene senza
stop), mentre la strategia è investita solo ~28% (è selettiva, resta in cassa). In un
toro ventennale questo divario di **esposizione**, da solo, spiega gran parte del
"vantaggio" del caso: non vince perché sceglie meglio, ma perché è **3× più sul mercato**.
Lo 0°–2° percentile era in larga parte un **artefatto dell'esposizione**, non un
fallimento della selezione.

Rendendo il gemello **via via più equo** (toggle `random_twin.match_exposure` per pari
esposizione, `manage_exits` per gli stessi stop della strategia):

| gemello | mediana CAGR | percentile strategia |
|---|---|---|
| classico (~92% investito, no stop) | +7.5% | **2°** |
| + pari esposizione (~28%) | +5.3% | **26°** |
| **+ stessi stop (test pulito)** | +4.2% | **59°** |

**Conclusione RIBALTATA (la correzione più importante del progetto):** quando il gemello
differisce dalla strategia **solo** nella selezione — stessa esposizione, stessi stop — la
strategia è al **59° percentile, SOPRA la mediana**. Weinstein (+4.5%) batte il caso
mediano (+4.2%). **La selezione di Weinstein aggiunge un valore positivo — modesto ma
reale.** Lo storico "0° percentile / la selezione non vale nulla" era un **artefatto di un
metro ingiusto**: confrontava un Weinstein selettivo e in cassa contro un caso pienamente
investito e senza stop. Corretti i due confondenti (esposizione + uscite), il segno si
inverte.

Onestà sui limiti: l'edge è **modesto** (59°, ~+0.3 punti sulla mediana), da **una** misura
(100 sorteggi, intervallo ampio [+2.8%, +7.3%]) → servono più sorteggi/periodi per esserne
sicuri. E resta sotto il benchmark (+4.5% vs +8.8% SPY: market timing, asse diverso).

> **NB metodologico:** il vecchio metro (gemello pienamente investito, senza stop) NON è un
> test pulito della selezione — confonde *cosa compri* con *quanto sei investito* e *come
> gestisci l'uscita*. Per il confronto corretto usa `match_exposure` + `manage_exits`.

Nuovi strumenti: `make_charts.py` (grafici, con entry/uscita reale/scarti col motivo),
`replot_cases.py` (replot veloce senza rifare il backtest). Nuovi toggle in config:
`base_volume_max_ratio` (contrazione volume nella base), `max_breakout_stretch` (scarta i
breakout da notizia con gap eccessivo oltre la resistenza — es. BIIB +34%: in automatico
si ignora, in uno scanner andrebbe segnalato al giudizio umano), `delisting_value`
(default `last_price`), `stop_intraweek: false`.

## SHORT, ESPOSIZIONE, RIBILANCIAMENTO — testati (luglio 2026), tutti OFF di default

Implementati come mirror del long e testati a fondo. **Nessuno migliora la selezione;**
il config resta sul **long-only pulito (59° percentile)**. Tutti disattivabili.

- **SHORT** (`short.enabled`, default OFF). Portafoglio long+short: long in Fase 2, short
  in Fase 4 (breakdown del supporto di un top), con volumi speculari (`top_vol_ratio`,
  `top_obv`), stop sopra l'entry, trailing MA30 che scende, `ma_breakup`, flip long↔short,
  costo di prestito (`borrow_annual`, ~8% realistico da Saxo). **Risultato: lo short NON
  paga** — hit rate 26-30% qualunque filtro (regime Fase 4, `bear_confirm_weeks`, volumi).
  In un toro ventennale i breakdown rimbalzano; confermare l'orso fa shortare più tardi,
  dentro il bear-rally (hit rate PEGGIORA a 26%). Long+short: CAGR +3.3% / 32° (vs +4.5% /
  59° del solo long).
- **ESPOSIZIONE** (sweep `risk_per_trade` × `max_position_pct`). La strategia è **signal-
  limited**: l'esposizione si ferma a ~52-54% anche a rischio 6%, mancano i segnali. Il CAGR
  ha un picco a ~+8.8% (= benchmark) al livello "medio" (2%/40%) poi CALA per concentrazione.
  Il meglio è **pareggiare il mercato con pura beta**, non alpha.
- **RIBILANCIAMENTO** (`portfolio.rebalance_on_signal`, default OFF). Se pienamente investito
  e arriva un segnale long, trimma le posizioni per includerlo. **Il turnover extra non
  ripaga** (peggiora il risultato).

Nuovi moduli/colonne: `Portfolio` gestisce short (`open_short`, `reduce`, `charge_borrow`,
`side`); `top_features` ha i volumi (`top_vol_ratio`, `top_obv`); schema cache **v7**.
Nuovi toggle: `short.*`, `bear_confirm_weeks`, `rebalance_on_signal`, `random_twin.
match_exposure`/`manage_exits`, `trailing_mode: ma30` + `ma_trail_atr_mult`, `base_obv_min`.
