# Notifiche Telegram

A ogni run, se compaiono **nuovi segnali PIENI**, ricevi un messaggio Telegram con
la lista + il link alla pagina. Solo i nuovi → niente spam.

## Setup (una volta)

### 1. Crea il bot
Su Telegram cerca **@BotFather** → `/newbot` → scegli un nome (es. "Weinstein
Screener") e uno username che finisca in `bot` (es. `dan_weinstein_bot`).
BotFather ti dà un **token** tipo `1234567890:AAE...`. Copialo.

### 2. Trova il tuo chat_id
- Apri una chat col **tuo bot** appena creato e mandagli un messaggio qualsiasi
  (es. "ciao").
- Poi visita nel browser (sostituisci il token):
  ```
  https://api.telegram.org/bot<IL_TUO_TOKEN>/getUpdates
  ```
  Cerca `"chat":{"id":XXXXXXXX,...}` → quel numero è il tuo **chat_id**.
- (In alternativa: scrivi a **@userinfobot**, ti risponde col tuo id numerico.)

### 3. Aggiungi i GitHub Secrets
Nel repo `weinstein-screener` → Settings → Secrets and variables → Actions:
- **`TELEGRAM_TOKEN`** = il token del bot (punto 1)
- **`TELEGRAM_CHAT_ID`** = il tuo chat_id (punto 2)

Da terminale, in alternativa:
```bash
gh secret set TELEGRAM_TOKEN
gh secret set TELEGRAM_CHAT_ID
```

## Prova subito che funziona
La notifica del job scatta solo sui **nuovi** pieni (potrebbero non essercene).
Per un test immediato, manda un messaggio di prova (sostituisci token e chat_id):
```bash
curl -s "https://api.telegram.org/bot<TOKEN>/sendMessage" --data-urlencode "chat_id=<CHAT_ID>" --data-urlencode "text=Test screener OK"
```
Se ti arriva "Test screener OK" su Telegram, token e chat_id sono giusti.

## Come funziona nel job
Lo step `notify.py` gira a fine run: legge `signals_log.json`, prende i pieni
**aggiunti in questa esecuzione** (`first_logged` = oggi) e, se ce ne sono, manda
il messaggio. Se i due secret non ci sono, non fa nulla (silenzioso).

## Tarature possibili (dimmi se le vuoi)
- Mandare anche un **heartbeat settimanale** ("nessun nuovo segnale") per sapere che il job è girato.
- Includere anche i **near-miss** (quasi) più forti.
- Notificare quando un **titolo seguito** (watchlist) colpisce lo stop o supera una soglia.
