-- Tabella della watchlist: i candidati "quasi" che segui a mano.
-- Chiave (ticker, settimana del segnale): seguire due volte lo stesso aggiorna.
CREATE TABLE IF NOT EXISTS follows (
  ticker      TEXT NOT NULL,
  signal_date TEXT NOT NULL,
  market      TEXT,
  entry       REAL,
  stop        REAL,
  base_len    INTEGER,
  mansfield   REAL,
  vol_ratio   REAL,
  currency    TEXT,
  added_at    TEXT,
  PRIMARY KEY (ticker, signal_date)
);

-- Iscritti al recap settimanale: chi fa /start al bot Telegram.
CREATE TABLE IF NOT EXISTS subscribers (
  chat_id  TEXT PRIMARY KEY,
  name     TEXT,
  added_at TEXT
);
