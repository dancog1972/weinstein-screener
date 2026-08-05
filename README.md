# Weinstein Bot — deliverable 1: motore di analisi delle fasi + backtest

Scanner e backtest del metodo delle 4 fasi di Stan Weinstein (analisi
settimanale: MA a 30 settimane, forza relativa Mansfield, volume sul
breakout, filtro di mercato).

## Avvio rapido (dati sintetici, senza API)
```bash
pip install -r requirements.txt
python -m pytest tests/          # 10 test del motore
python run.py backtest           # genera output/report.html
```
Apri `output/report.html` in Chrome: è autonomo e funziona offline.

## Passaggio a dati reali (EODHD)
1. Sottoscrivi il piano "EOD Historical Data — All World" su eodhd.com
2. `export EODHD_API_KEY=la_tua_chiave`
3. In `config.yaml`: `provider: eodhd`, compila `universe` con i ticker
   (formato EODHD: `AAPL.US`, `MC.PA`, `SAP.XETRA`) e i benchmark
   (`GSPC.INDX` per gli USA, `STOXX.INDX` per l'Europa)
4. `python run.py fetch` (scalda la cache Parquet), poi `python run.py backtest`

## Struttura
- `config.yaml` — TUTTI i parametri del metodo (soglie, filtri, costi)
- `src/providers/` — fonti dati intercambiabili (synthetic, eodhd)
- `src/indicators.py` — ricampionamento settimanale, MA30, Mansfield RS, volumi
- `src/stages.py` — classificazione delle 4 fasi (funzione pura)
- `src/signals.py` — rilevamento del breakout di Fase 2 con tutti i filtri
- `src/portfolio.py` — sizing a rischio fisso, costi, registro operazioni
- `src/backtest.py` — loop settimanale, metriche, split in/out-of-sample
- `src/report.py` — dashboard HTML standalone con config incorporata

## Il ciclo di taratura (approccio A)
Il report incorpora la config del run: modificala nel pannello in fondo,
scaricala e rilancia `python run.py backtest --config config_modificata.yaml`.
Confronta sempre in-sample vs out-of-sample per non sovra-ottimizzare.

## Garanzie anti look-ahead
- Ogni indicatore alla settimana t usa solo dati fino a t (il volume di
  confronto usa le 4 settimane PRECEDENTI, `shift(1)`)
- Il segnale nasce alla chiusura del venerdì; l'esecuzione simulata applica
  slippage; gli stop sono valutati sulla chiusura settimanale successiva
- La resistenza della base è il massimo dei massimi FINO alla settimana
  precedente il breakout

## Limiti dichiarati della v1
- Provider synthetic: i numeri validano il MOTORE, non il metodo
- Filtro settoriale predisposto ma disattivo (servono proxy settoriali)
- Trailing "da trader" implementato: exits.trailing_swing (pivot_k, swing_buffer)
- Prossimo deliverable: simulatore interattivo human-vs-bot (FastAPI,
  modalità trasparente e cieca, sizing manuale per il giocatore)

## Deliverable 2 — Simulatore human-vs-bot (FastAPI)

Gioca la storia settimana per settimana contro il bot: tu decidi gli ingressi
(accetta/passa, con taglia manuale), il bot applica le regole; le uscite
seguono le stesse regole per entrambi. Alla fine si confrontano le curve.

```bash
pip install -r requirements.txt
python play.py --config config_sim.yaml      # apre su http://localhost:8000
```

- **Modalità cieca** (default): ticker e date anonimizzati ("STOCK-03",
  "settimana 142") — misura il tuo giudizio chartistico senza memoria storica.
- **Modalità trasparente**: ticker e date reali, utile per imparare il metodo.
- Il grafico è sempre TAGLIATO al presente (nessun futuro visibile).

`config_sim.yaml` usa dati sintetici con soglie "da palestra" (volume 1.5×,
market filter off) per avere abbastanza eventi da giocare. NON sono soglie
realistiche: servono a rodare la meccanica, non a validare la strategia.
Quando avrai l'abbonamento, lo stesso simulatore gira su dati reali cambiando
solo il provider in config.

Nota: in questa v1 le uscite non sono discrezionali (regole identiche per
umano e bot); il gioco è sulla scelta e il sizing degli INGRESSI. La gestione
manuale delle uscite è una possibile estensione futura.

### Ordini: a mercato o buy limit

Nella scheda del segnale scegli tra:
- **A mercato**: entri alla chiusura del breakout (+ slippage), come il bot.
- **Buy limit**: piazzi un ordine sotto il prezzo, per prendere il pullback.

Regole del limite (in `config_sim.yaml`, sezione `simulator`):
- vive `limit_weeks` settimane (default 4), poi **scade**;
- si esegue se il **minimo** di una settimana successiva tocca il limite
  (riempire sulla chiusura sarebbe barare: il minimo può rimbalzare);
- viene **annullato** se il setup si invalida — prezzo che chiude sotto lo
  stop, o rottura della MA30 in discesa;
- non è valutato nella settimana in cui lo piazzi (altrimenti si
  autoeseguirebbe sul breakout stesso).

Il trade-off è reale e il simulatore te lo fa toccare: il limite dà un
ingresso migliore e uno stop più stretto, ma **ti fa perdere i breakout più
forti**, quelli che partono e non tornano mai indietro. È il punto dove il
tuo giudizio può battere il bot, che compra sempre a mercato.

Il pannello mostra sempre cassa libera, cassa impegnata dagli ordini,
percentuale investita e posizioni aperte su massimo; mentre digiti quantità
e prezzo vedi **valore dell'ordine** e **rischio in euro** (quantità ×
distanza dallo stop), con avviso se superi la cassa disponibile.

### Tre modalità di stop (solo per il giocatore umano)

Il bot usa sempre il suo stop (buffer sotto la resistenza). Tu scegli tra:

1. **Come il bot** — buffer percentuale sotto la resistenza rotta.
2. **Minimo relativo** — l'ultimo pivot low CONFERMATO prima del breakout:
   l'ancora prescritta da Weinstein, il primo supporto vero sotto il prezzo.
   Non è il fondo della base (su una base lunga è lontanissimo e irrilevante).
   Se nessun pivot è confermato, l'opzione è disabilitata: non si inventa.
3. **Discrezionale** — tra 5% e 10% sotto il prezzo di ingresso.

**Il sizing si adatta da solo.** Con rischio fisso all'1% dell'equity,
`quantità = rischio in euro ÷ (ingresso − stop)`. Quindi uno stop più lontano
significa **meno azioni**, non più rischio. Esempio reale dal simulatore:

| stop | prezzo | rischio/azione | quantità suggerita |
|------|--------|----------------|--------------------|
| bot | $93.44 | 4.3% | 205 |
| minimo relativo | $84.17 | 13.8% | 74 |
| discrezionale | $90.34 | 7.5% | 136 |

Stesso rischio in euro, posizioni molto diverse. Lo stop largo non ti fa
stoppare dal rumore, ma la posizione più piccola guadagna meno se il titolo
corre. Lo stop stretto compra di più, ma viene colpito dai falsi movimenti.
Il punto giusto è dove la STRUTTURA dice che il setup è invalidato.

Il pannello mostra sempre valore ordine e rischio in euro/percentuale mentre
scegli, avvisa se superi la cassa libera, e segnala quando piazzi lo stop
sopra il minimo relativo (dentro la zona di respiro del titolo). Il report
finale riporta hit rate e rendimento medio **per modalità di stop usata**.

### Export della partita

Il pulsante **Esporta** (durante il gioco) e **Scarica partita** (a fine
partita) salvano un JSON con tutto ciò che serve a rivedere e analizzare:

- `journal` — ogni segnale proposto con il **contesto che avevi davanti**
  (volume ratio, Mansfield, base, le opzioni di stop, equity e cassa al
  momento della scelta) e cosa avete deciso tu e il bot. Senza il contesto,
  rileggendo la partita non capiresti *perché* avevi deciso così.
- `trades` — operazioni chiuse di entrambi, con motivo di uscita, rendimento
  e **quale modalità di stop** avevi usato.
- `divergences` — la parte che conta: i segnali che hai passato e il bot ha
  preso (con l'esito, quindi quanto ti è costato il tuo filtro), quelli che
  hai preso da solo, e quelli su cui eravate d'accordo.
- `events`, `equity_curves`, `meta` (config, modalità cieca, avviso se i dati
  sono sintetici).

C'è anche `Operazioni (CSV)` per aprire i trade in Excel.

Il JSON è il file da conservare tra una sessione e l'altra, o da passare a
qualcuno per farsi analizzare le decisioni.

## Backtest ampio su universo storico (S&P 500 point-in-time)

### Perché point-in-time
Testare i 500 componenti ATTUALI dell'S&P su 20 anni significa scegliere
aziende che sappiamo essere sopravvissute: Enron, Lehman e centinaia di nomi
minori spariscono dal campione. Per una strategia di breakout questo gonfia i
risultati, perché molti falsi breakout storici sono avvenuti su titoli poi
morti. Il modulo `src/universe.py` risolve il problema: **un titolo è
comprabile solo nelle settimane in cui era davvero nell'indice**, delistati
inclusi fino alla loro uscita.

### Prerequisito: il piano giusto
I componenti storici stanno nel pacchetto **Fundamentals** di EODHD, non nel
piano "EOD Historical Data" base. Verifica sul sito quale piano include
l'endpoint `fundamentals/GSPC.INDX?filter=HistoricalTickerComponents`.
Copertura: S&P 500 dal 2000, ~770 ticker storici con date di ingresso/uscita
e flag `IsDelisted`.

### Procedura
```bash
export EODHD_API_KEY=la_tua_chiave

# 1) composizione storica (1 chiamata)
python build_universe.py constituents --index GSPC.INDX

# 2) prezzi di tutti i ticker mai apparsi (~800 chiamate, ~10 min)
python build_universe.py prices --index GSPC.INDX --start 2000-01-01

# 3) backtest
python run.py backtest --config config_sp500.yaml
```
Lo scaricamento è **idempotente e resiliente**: un ticker fallito non ferma
gli altri, finisce in `data/failed_tickers.json`, e rilanciando lo stesso
comando si ritenta solo ciò che manca. Ctrl+C non perde il lavoro.

Il report dichiara in testa che l'universo è point-in-time e quanti segnali
sono stati scartati perché il titolo non era nell'indice a quella data.

### Europa: il limite dei dati
EODHD fornisce composizione storica **solo per indici S&P/Dow Jones**. Per
STOXX 600 e indici nazionali esiste solo la composizione ATTUALE:
```bash
python build_universe.py current --index STOXX.INDX   # genera la lista YAML
python run.py backtest --config config_europe.yaml    # mode: index_static
```
Con `mode: index_static` il report stampa in testa un avviso: i risultati sono
un **limite superiore** della performance reale. Usa l'Europa come *validazione
di robustezza* del risultato USA, non come test primario.

### Come leggere i risultati
Non tarare i parametri guardando l'in-sample. Guarda la colonna
**out-of-sample**: se l'edge sparisce lì, era overfitting. E confronta sempre
con il benchmark buy & hold: la letteratura sul trend-following documenta
soprattutto una **riduzione dei drawdown**, non rendimenti superiori.

## Backtest ampio col piano da €19.99 (universo per liquidità)

### L'indice non serve
L'indice serviva a due cose: (1) benchmark per la Mansfield RS e il filtro di
regime, (2) definire l'universo. Per (1) basta la **serie prezzi** di `SPY.US`,
inclusa nel piano base. Per (2) c'è un'alternativa migliore.

`mode: liquidity` definisce l'universo con una **regola sui dati**: prezzo
minimo, volume medio minimo, turnover in dollari minimo — tutti calcolabili
dai dati EOD che già scarichi, **settimana per settimana**. Un titolo entra
quando diventa liquido, esce quando smette o muore. Il survivorship bias è
eliminato dallo stesso meccanismo, senza pagare i Fundamentals.

È anche metodologicamente superiore per Weinstein: l'S&P 500 contiene solo
mega-cap, che passano anni in Fase 2 senza formare basi (l'abbiamo verificato
su AAPL/AMZN: 4-7 rotture in 12 anni). Le mid-cap liquide sono il terreno
naturale del metodo. E una soglia di liquidità è **replicabile**, mentre la
composizione di un indice è la decisione di un comitato.

Nota: la capitalizzazione richiederebbe il numero di azioni (Fundamentals).
Usiamo il **turnover in dollari** come proxy — e per uno screener è persino
più pertinente: ciò che conta è entrare e uscire senza muovere il prezzo.

### Il triple screen di Weinstein
`sector_filter: true` attiva il secondo schermo: **mercato → settore → titolo**.
La forza del settore è misurata sugli **11 ETF settoriali SPDR** (XLK, XLF,
XLV...) con lo stesso motore di fasi usato per i titoli. Un segnale passa solo
se il suo ETF settoriale è in Fase 2 con RS positiva.

Due limiti dichiarati: la mappa ticker→settore (`data/sector_map_us.csv`) è
**statica** (classificazione di oggi applicata alla storia); e XLRE esiste dal
2015, XLC dal 2018 — prima di quelle date il codice usa il settore predecessore
(Real Estate stava nei Financials, Communication Services in Technology).
Titoli non mappati: filtro **inattivo**, mai scarto arbitrario.

### Procedura
```bash
export EODHD_API_KEY=la_tua_chiave

# 1) lista ticker USA, attivi + delistati, filtro grossolano (2 chiamate)
python build_universe.py screen --exchange US --limit 800

# 2) prezzi + ETF settoriali (~810 chiamate, ~15 min)
python build_universe.py prices --from-universe data/universe_US.parquet \
    --start 2000-01-01 --with-sector-etfs

# 3) backtest col triple screen
python run.py backtest --config config_us_liquidity.yaml
```

`--limit` serve solo per prove: un limite arbitrario reintroduce una selezione
non replicabile. Per il backtest vero, toglilo (ma preparati a molte chiamate).

Il report dichiara in testa il numero di membri per settimana, i segnali
scartati per liquidità e quelli scartati per settore debole.

### Confronto da fare
Lancia il backtest **con e senza** `sector_filter`, e guarda l'out-of-sample.
Il triple screen è nel libro, ma ogni filtro riduce i segnali e aumenta il
rischio di overfitting. Se paga, si vedrà nei dati; se non paga, lo saprai
invece di assumerlo.

## Universo per liquidità + triple screen (piano EODHD €19.99)

### Perché non serve l'indice
L'indice serve a due cose: fare da **benchmark** (per la RS di Mansfield e il
filtro di regime) e **definire l'universo**. Solo la prima è indispensabile, e
si risolve con la serie prezzi di `SPY.US` — nessun dato fondamentale.

Per l'universo usiamo una **regola** invece dell'appartenenza a un indice:
prezzo minimo, volume medio, turnover medio. È metodologicamente superiore.

**Anti-lookahead per costruzione.** Le soglie si valutano su medie che
finiscono alla settimana *precedente* (`shift(1)` su tutti e tre i criteri,
prezzo incluso). Senza lo shift, un titolo che sfonda i $5 o esplode di volume
proprio nella settimana del breakout diventerebbe idoneo *grazie al breakout
stesso*. Un titolo liquido nel 2005 entra nell'universo del 2005; se muore nel
2008, esce nel 2008: **il survivorship bias sparisce gratis**.

**Cattura i titoli giusti.** L'S&P 500 contiene mega-cap che passano anni in
Fase 2 senza formare basi. Weinstein cerca chi *esce* da una base: le mid-cap
liquide sono il terreno naturale del metodo.

### Il triple screen: mercato → settore → titolo
Weinstein ripete "the forest before the trees". Il secondo schermo è il settore,
misurato sugli **ETF settoriali SPDR** (XLK, XLF, XLE…) con lo stesso motore di
fasi usato per i titoli — dati di prezzo reali, storici, nessun senno di poi.

Due dettagli che contano. Gli ETF nati dopo (XLRE nel 2015, XLC nel 2018) hanno
un **predecessore**: prima dello scorporo, il Real Estate stava in XLF e le
Communication Services in XLK. E se un titolo non è mappato, o l'ETF non esiste
ancora, il filtro è **inattivo** — non esclude arbitrariamente.

### Procedura
```powershell
$env:EODHD_API_KEY = "la_tua_chiave"
python build_universe.py screen --exchange US --limit 800
python build_universe.py prices --from-universe data/universe_US.parquet `
       --start 2000-01-01 --with-sector-etfs
python run.py backtest --config config_us_liquidity.yaml
```

### Bias residui, dichiarati nel report
- Il **turnover** (prezzo × volume) è un proxy della dimensione, *non* la
  capitalizzazione: quella richiede il numero di azioni, dato a pagamento.
  Per uno screener operativo è probabilmente più pertinente.
- La mappa ticker→settore è **statica**: la classificazione odierna applicata
  alla storia. Bias lieve ma reale (Amazon è stata "Retail" prima che "Tech").

### Confronta, non assumere
Ogni filtro riduce i segnali e aumenta il rischio di overfitting. Lancia il
backtest con `sector_filter: true` e `false` e guarda l'**out-of-sample**. Se
il triple screen paga, si vedrà. Se non paga, lo saprai — invece di assumerlo
perché lo dice il libro.

## Volatilità invece di capitalizzazione

Weinstein preferisce società di una certa dimensione perché danno **"buona
liquidità e variazioni contenute"**. Ma la capitalizzazione è il *proxy*, non
l'obiettivo — e gli obiettivi sono entrambi misurabili direttamente:

| ciò che Weinstein vuole | proxy classico | misura diretta |
|---|---|---|
| buona liquidità | capitalizzazione | **turnover** (prezzo × volume) |
| variazioni contenute | capitalizzazione | **ATR%** (ATR ÷ prezzo) |

Il turnover è persino *migliore* della cap per la liquidità: una large-cap poco
scambiata è illiquida nonostante la dimensione, e il turnover se ne accorge.

E c'è un argomento decisivo contro la cap. Un'API gratuita ti dà la
capitalizzazione **di oggi**. Applicarla al 2005 sarebbe **look-ahead**: il
filtro selezionerebbe le small-cap del 2005 *proprio perché sappiamo* che sono
diventate mega-cap. Peggio del survivorship bias. La cap storica vera
(prezzo(t) × azioni(t)) richiede lo shares outstanding storico, che le API
gratuite non hanno — men che meno sui delistati, che sono metà del valore del
lavoro fatto.

```yaml
liquidity:
  min_dollar_volume: 5000000
  max_volatility: 0.08    # ATR settimanale ≤ 8% del prezzo (null = disattivo)
  atr_weeks: 14
```

Anche qui **shift(1)**: la barra del breakout è per definizione ampia, e senza
lo shift escluderebbe il titolo proprio nella settimana del segnale.

### Il report risponde con i numeri
Il filtro di volatilità aggiunge selettività, o il turnover lo rendeva già
ridondante? Domanda empirica. Il report stampa la **correlazione
turnover↔volatilità** e la quota di settimane escluse *solo* dalla volatilità.

Su un universo realistico simulato la correlazione è ≈ **−0.92** (i titoli molto
scambiati sono davvero i più tranquilli), ma il filtro esclude comunque titoli
con turnover ampio e ATR% alto: mid-cap nervose che passavano il primo criterio.
Sono esattamente i falsi breakout che Weinstein voleva evitare.

Se un giorno vorrai la capitalizzazione vera, servono i Fundamentals per un mese
(scarichi lo shares outstanding storico, metti in cache, disdici). Ma fallo
**dopo** aver visto se la volatilità basta — non prima.

## Nota sui ticker EODHD (bug corretto in v3.2.2)

EODHD distingue due cose che sembrano la stessa:
- il **codice di richiesta** dell'API: `US` — copre tutte le borse americane
- la **borsa reale** del titolo: `NYSE`, `NASDAQ`, `BATS`, `OTC`...

L'endpoint `exchange-symbol-list/US` restituisce un campo `Exchange` che
contiene la *seconda*. Costruire il ticker con quella produce `AAPL.NASDAQ`,
che l'API non conosce: **404 su ogni chiamata**. Il ticker corretto è `AAPL.US`.

Il codice ora conserva entrambi: `ticker` (per l'API) e `exchange_real` (per
escludere l'OTC).

**Sicurezza:** i messaggi d'errore di `requests` includono l'URL completo, che
contiene `api_token=...`. Ora la chiave viene mascherata prima di finire in
`failed_tickers.json`.

## Revisione del codice (v3.4): cinque bug trovati

Audit sistematico prima del primo backtest sui dati veri.

**1. Doppio minimo ignorato (grave).** `confirmed_pivot_lows` esigeva che tutte
le barre della finestra fossero *strettamente* sopra il pivot. Un doppio minimo
— il double bottom, formazione centrale in Weinstein — non produceva alcun
pivot. Conseguenza: trailing stop che non salivano, stop "minimo relativo"
assenti proprio dove la struttura è più significativa.

**2. Prezzi su scale diverse (grave).** `base_features` calcolava resistenza e
minimo della base su `high`/`low` GREZZI, mentre il segnale confronta
`adj_close > resistance`. Su AAPL (split 7:1 nel 2014) il close grezzo pre-split
vale ~$650 e l'adj_close ~$93: due scale incompatibili, e il breakout non scatta
mai. Ora `enrich_weekly` calcola `adj_high`/`adj_low` con lo stesso fattore.

**3. Posizioni fantasma sui delistati (gravissimo).** Un titolo che esauriva i
dati restava in portafoglio *per sempre*, valutato all'ultimo prezzo noto e
occupando uno slot di `max_positions`. Con l'universo per liquidità, dove due
terzi dei titoli sono delistati, l'equity finale sarebbe stata piena di aziende
morte. Ora la posizione viene chiusa d'ufficio (`reason: delisting`) e il titolo
sparisce dalla valutazione.

**4. Equity che nasconde le perdite.** `prices.get(tk, entry_price)` valutava una
posizione senza prezzo al costo d'acquisto. Ora si usa `last_price`.

**5. Ri-download infinito.** Il warmup risale 80 settimane prima di `start`: con
`start: 2001-06-01` si arriva al 1999-11-19, mentre i dati partono dal 2000. La
cache risultava sempre insufficiente e **l'intero universo veniva riscaricato a
ogni backtest**. Ora un file `.meta.json` registra da quale data i dati furono
chiesti.

### Migrazione della cache esistente
```powershell
python migrate_cache.py --start 2000-01-01
```
Da lanciare una volta sola sulla cache già scaricata, altrimenti il primo
backtest riscaricherebbe tutti gli 825 ticker.

## Diagnostica del funnel

Un backtest che produce zero segnali non è un fallimento: è un'informazione.
Ma serve sapere **quale filtro** li ha uccisi.

```powershell
python diagnose.py --config config_us_liquidity.yaml
```

Scompone il funnel condizione per condizione e stampa quante settimane-titolo
sopravvivono a ciascuna. Se il conteggio si azzera su `volume_sufficiente`,
la soglia `volume_ratio_min: 2.0` (che viene dal 1988) è troppo alta per i
mega-cap moderni. Se si azzera su `liquidita`, le soglie dell'universo sono
strette. Se su `mercato_favorevole`, il filtro di regime esclude gran parte
del periodo.

## Taratura della volatilità: un avvertimento

Il default `max_volatility` è ora **null (disattivato)**, e non per pigrizia.

Calibrando la soglia sui dati *sintetici* avevo scelto 8%, perché lì l'ATR%
mediano è ~4%. Su un universo USA realistico l'ATR% mediano dei titoli che già
passano prezzo+volume+turnover è circa **17%**. Con la soglia all'8% il funnel
crolla dell'85% e il backtest resta senza segnali.

| soglia | settimane idonee tagliate |
|---|---|
| 8% | ~85% |
| 12% | ~70% |
| 15% | ~57% |
| 20% | ~35% |

**Parti disattivato.** Guarda quanti segnali produce l'universo. Poi, se e solo
se i falsi breakout sono un problema misurabile nei risultati, attiva partendo
da 0.20 e stringi. È la lezione generale: una soglia tarata su dati simulati
non sopravvive ai dati veri.

## Sanificazione dei dati (v3.7)

I dati EOD grezzi contengono errori che a valle diventano "breakout" fantasma
con rendimenti a quattro cifre e avvelenano ogni media aggregata:

- prezzi a **zero o negativi** (viste righe con entry 0.00);
- **split-adjustment corrotto**: quando `close≈0`, il fattore `adj_close/close`
  esplode e trascina le resistenze a valori come 1.000.000;
- **volume ratio infinito**: media del volume precedente pari a zero;
- **Mansfield fuori scala** (>500%): benchmark o prezzo corrotto in quel punto.

`DataProvider.validate` ora scarta queste righe alla fonte, e un titolo con
>50% di barre corrotte viene escluso del tutto. La sanificazione gira anche in
**lettura dalla cache**, così i dati già scaricati vengono ripuliti senza
riscaricare. Una seconda rete in `detect_signals` scarta i segnali con valori
non finiti o fuori scala che dovessero sfuggire.

Questo NON è ottimizzazione della strategia: è igiene dei dati. Senza, qualunque
statistica aggregata è inaffidabile.

## Velocità: la trappola dei delistati (v3.7.1)

Il primo backtest sull'universo completo ha impiegato ~3 ore. Il calcolo puro
è ~8 minuti e l'I/O ~1: le altre ~2h50 erano **chiamate API inutili**.

Causa: due terzi dell'universo sono delistati, con dati che finiscono alla loro
morte (2015, 2008...). Il backtest chiede dati fino al 2024, la cache di un
delistato arriva al 2015, il codice la giudicava "incompleta" e richiamava
l'API a ogni run — a caccia di dati che non esistono.

Correzione: la cache è completa se abbiamo GIÀ CHIESTO fino a `end` (campo
`asked_to` nel meta), anche se i dati finiscono prima. Un delistato non "manca
di dati recenti": è morto. La correzione è precisa — un titolo con cache
davvero incompleta viene ancora aggiornato.

La cache del settimanale arricchito (già presente) dà un ulteriore **13.7×**:
dopo il primo run, i successivi ricalcolano solo se cambiano i parametri di
stage o il benchmark. Cambiare `volume_ratio_min` o gli stop riusa la cache.
Risultato: la sperimentazione costa minuti, non ore.

## Revisione del codice (v3.8) — bug trovati e corretti

Revisione sistematica pre-benchmark. Tre correzioni, tutte nella direzione
dell'onestà (nessuna gonfiava i risultati a favore).

**1. Look-ahead di esecuzione (il più importante).** Gli ingressi avvenivano
alla CHIUSURA della settimana del breakout — lo stesso prezzo che rivelava il
segnale, impossibile da ottenere nella realtà. Ora l'esecuzione avviene
all'apertura della settimana SUCCESSIVA (`execution: next_open`, default), cioè
ciò che potresti fare davvero vedendo il breakout a mercato chiuso. Se il gap
di apertura è già sotto lo stop, il trade si salta. Questo rende i risultati
un po' più severi, ma onesti.

**2. Split OOS sovrapposto.** `equity.loc[:split]` e `equity.loc[split:]` sono
entrambi inclusivi in pandas: la settimana esatta dello split finiva in
entrambi i periodi. Ora l'out-of-sample parte strettamente dopo lo split.

**3. Delisting ottimistico (dichiarato, non corretto).** La chiusura per
delisting avviene all'ultimo prezzo noto: realistico per fusioni, ottimistico
per i fallimenti. Il piano dati non dà il motivo del delisting, quindi la stima
è un limite superiore su quei casi. Ora è dichiarato nel report.

Invarianti verificati e CORRETTI (nessun bug): niente look-ahead negli
indicatori (test di troncamento), survivorship bias eliminato (i delistati
escono, non congelano il prezzo), costi applicati a entrambi i lati (15 bps/lato),
sizing a rischio costante anche col gap di esecuzione, market filter coerente
(valuta alla chiusura di t, come il breakout).

## v3.9.2 — gemello vettorizzato (da ore a minuti)

**Gemello casuale** (`random_twin` nel config). Risponde alla domanda vera:
la SELEZIONE Weinstein batte lo scegliere a caso? Stesso universo, capitale,
costi, vincoli e market filter; sceglie a caso invece che coi segnali, e non
usa stop (esce solo quando il titolo lascia l'universo). Ripetuto su n_sims
semi → distribuzione. La strategia sta al 95° percentile (picking eccellente)
o al 50° (uguale al caso)?

**Esperimento sul rischio** (`python run.py experiment`). La stessa strategia
a rischi diversi (1%, 2%, 5%) in un solo lancio, cache condivisa. Se salendo
di rischio il CAGR non cresce ma il MaxDD peggiora, l'edge è negativo e la leva
amplifica solo le perdite — lo si vede coi numeri invece che a intuito.

**Stop realistico** (`stop_intraweek: true`). Lo stop è valutato sul minimo
settimanale, non sulla chiusura: un ordine reale scatta quando il prezzo lo
tocca, non aspetta il venerdì. Più severo e più onesto. Esce allo stop (o
all'apertura se la settimana ha aperto in gap sotto).

**Tetti rimossi**: niente max_positions né max_position_pct — si apre finché
c'è cash, come richiesto. Attenzione: con rischio alto e stop stretti, una
singola posizione può assorbire gran parte del capitale.

## v3.9.2 — gemello casuale vettorizzato

Il gemello ingenuo richiamava is_eligible per ogni titolo, ogni settimana, ogni
sorteggio: ~1 miliardo di operazioni su 100 sorteggi = ore (bloccava a twin
20/100). Ora il pannello prezzi+idoneità si pre-calcola UNA volta in array
NumPy e si condivide fra tutti i sorteggi: ~16 milioni di operazioni.

Misurato alla scala reale (1230 settimane × 8000 titoli idonei): **2-3 minuti
per 100 sorteggi**, contro 5+ ore. Risultati identici alla versione lenta.

## v3.9.3 — revisione completa (Fable 5)

**Bug grave corretto nel gemello**: i delistati venivano rimborsati al prezzo
di ENTRATA (perdite cancellate) — survivorship bias dentro il metro che doveva
esserne immune. Con 2/3 dell'universo delistato, gonfiava il rendimento del
caso e faceva sembrare il picking peggiore del reale. Ora si liquida
all'ultimo prezzo valido. Test di regressione dedicato.

**Correzione esatta su cash esaurito**: il costo per posizione del gemello è
identico per ogni candidato (shares·p non dipende da p), quindi quando il cash
non basta per uno non basta per nessuno: break, non continue.

**Verificati e corretti** (nessun altro bug): duplicati bloccati da can_open,
esecuzione next_open coerente con lo stop intrasettimanale, split OOS
disgiunto, costi su entrambi i lati, sizing a rischio costante.

## v3.10 — esperimento multi-rischio in una passata

Prima, `experiment` rifaceva l'intera fase pesante (scansione 20.000 titoli +
segnali) per ogni livello di rischio: 3 rischi = 3× il lavoro pesante. Ora la
fase pesante si calcola UNA volta (`_prepare_backtest`) e si riusa per tutti i
rischi; solo la simulazione di portafoglio (leggera) si ripete.

Misurato: rischi 2 e 5% passano da ~15 min a pochi secondi ciascuno. Il test
`test_prepared_reuse_gives_identical_results` garantisce che il riuso dia
risultati identici al calcolo da zero (verificato al 9° decimale).

## v3.11 — decomposizione del metodo (misura 1)

Rimosso l'esperimento multi-rischio (domanda chiusa: il rischio non crea
rendimento su un edge negativo) e il suo riuso buggato. Backtest singolo pulito.

Nuovo `analyze.py` — decomposizione di dove Weinstein aggiunge o distrugge
valore, una misura per volta.

**Misura 1 — qualità pura degli ingressi**: per ogni segnale, il rendimento
tenendo 4/13/26/52 settimane SENZA stop né filtri, assoluto e in eccesso sul
mercato. Se gli ingressi battono il mercato in isolamento, il valore si perde
a valle (stop, filtri); se perdono, è la selezione a non funzionare.

  python analyze.py --config config_us_liquidity.yaml --measure entries

## v3.14 — short in Fase 4 (segnali + misura 1-short)

Motore dei segnali SHORT completo e testato:
- `top_features`: simmetrico di base_features per i top (Fase 3), con supporto
  CONGELATO (stabilito prima della rottura) — senza, il prezzo non potrebbe mai
  romπερε il proprio minimo. Anti-lookahead verificato.
- `detect_signals_short`: breakdown sotto supporto, MA30 in discesa, Mansfield
  negativo, volume alto, mercato in Fase 4. Stop SOPRA l'ingresso.
- `confirmed_pivot_highs` / `last_confirmed_pivot_high_before` per lo stop short.

Misura 1-short (`analyze.py --measure entries-short`): i breakdown, tenuti
short, guadagnano quando il titolo scende più del mercato? Forward returns
ribaltati (per lo short, prezzo che scende = guadagno). È il test che decide
se vale la pena costruire il portafoglio short completo.

  python analyze.py --config config_us_liquidity.yaml --measure entries-short

## v3.15 — analisi del filtro Mansfield

Test dell'intuizione: un titolo che esce da una Fase 1 (declino + base) ha
Mansfield ancora negativo, perché la media a 52 settimane porta il peso del
declino. Il filtro 'mansfield >= 0' potrebbe escludere i migliori candidati —
quelli che stanno invertendo.

- Test A: confronta la qualità dei segnali con Mansfield positivo (tenuti dal
  filtro) vs negativo (scartati). Se gli scartati battono il mercato di più,
  il filtro danneggia.
- Test B: separa i segnali in 4 quadranti (livello × direzione del Mansfield).
  Se 'negativo ma in salita' batte 'positivo ma in calo', la direzione conta
  più del livello.

Reso disattivabile il filtro sulla direzione (require_mansfield_rising), per
permettere ai quadranti di vedere tutti i segnali.

  python analyze.py --config config_us_liquidity.yaml --measure mansfield
