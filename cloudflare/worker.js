// Worker Cloudflare: watchlist dei segnali "quasi" seguiti a mano.
// La pagina (pubblica su GitHub Pages) chiama questo Worker per seguire/smettere
// di seguire un candidato; il job settimanale legge la lista. Il DB è D1 (SQLite).
//
// SICUREZZA: la pagina è pubblica, quindi ogni scrittura richiede un SEGRETO
// (env WATCH_SECRET). Il segreto NON sta nella pagina: l'utente lo digita una
// volta nel browser (localStorage) e viaggia con ogni richiesta. Solo chi lo
// conosce può modificare la watchlist. CORS aperto (le scritture sono comunque
// protette dal segreto).

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

const json = (o, status = 200) =>
  new Response(JSON.stringify(o), {
    status,
    headers: { ...CORS, "Content-Type": "application/json" },
  });

const num = (v) => (v === null || v === undefined || v === "" || isNaN(Number(v)) ? null : Number(v));
const int = (v) => (num(v) === null ? null : Math.trunc(Number(v)));
const str = (v, n) => String(v ?? "").slice(0, n);

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { headers: CORS });
    const url = new URL(request.url);

    // --- LETTURA: GET /list?secret=... → la watchlist (per il job e per la pagina)
    if (request.method === "GET" && url.pathname === "/list") {
      if (url.searchParams.get("secret") !== env.WATCH_SECRET) return json({ error: "unauthorized" }, 401);
      const { results } = await env.DB.prepare(
        "SELECT ticker, signal_date, market, entry, stop, base_len, mansfield, vol_ratio, currency, added_at " +
        "FROM follows ORDER BY added_at DESC"
      ).all();
      return json(results || []);
    }

    // --- SCRITTURA: POST /follow oppure /unfollow (richiede il segreto nel body)
    if (request.method === "POST") {
      let b;
      try { b = await request.json(); } catch { return json({ error: "bad json" }, 400); }
      if (b.secret !== env.WATCH_SECRET) return json({ error: "unauthorized" }, 401);
      const tk = str(b.ticker, 40), d = str(b.date, 10);
      if (!tk || !d) return json({ error: "missing ticker/date" }, 400);

      if (url.pathname === "/follow") {
        await env.DB.prepare(
          "INSERT OR REPLACE INTO follows " +
          "(ticker, signal_date, market, entry, stop, base_len, mansfield, vol_ratio, currency, added_at) " +
          "VALUES (?,?,?,?,?,?,?,?,?,?)"
        ).bind(tk, d, str(b.market, 8), num(b.entry), num(b.stop), int(b.base_len),
               num(b.mansfield), num(b.vol_ratio), str(b.currency, 4), new Date().toISOString()).run();
        return json({ ok: true, following: true });
      }
      if (url.pathname === "/unfollow") {
        await env.DB.prepare("DELETE FROM follows WHERE ticker=? AND signal_date=?").bind(tk, d).run();
        return json({ ok: true, following: false });
      }
    }
    return json({ error: "not found" }, 404);
  },
};
