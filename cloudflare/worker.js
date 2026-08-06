// Worker Cloudflare: watchlist dei "quasi" seguiti + iscrizioni al bot Telegram.
//
// Rotte:
//   POST /follow, /unfollow   watchlist (segreto WATCH_SECRET nel body) — solo tu
//   GET  /list?secret=        la watchlist (per il job)
//   GET  /subscribers?secret= gli iscritti al recap (per il job)
//   POST /telegram            WEBHOOK Telegram: /start iscrive + benvenuto, /stop disiscrive
//
// SICUREZZA:
//   - le scritture watchlist e le letture (/list, /subscribers) usano WATCH_SECRET;
//   - il webhook /telegram è validato dall'header secret di Telegram (TG_WEBHOOK_SECRET).
// Gli AMICI non inseriscono nulla: fanno /start, il Worker li iscrive e risponde.

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};
const json = (o, s = 200) =>
  new Response(JSON.stringify(o), { status: s, headers: { ...CORS, "Content-Type": "application/json" } });
const num = (v) => (v === null || v === undefined || v === "" || isNaN(Number(v)) ? null : Number(v));
const int = (v) => (num(v) === null ? null : Math.trunc(Number(v)));
const str = (v, n) => String(v ?? "").slice(0, n);

function tg(env, method, body) {
  return fetch(`https://api.telegram.org/bot${env.TELEGRAM_TOKEN}/${method}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
}

function welcome(env) {
  const s = (env.SITE_URL || "").replace(/\/$/, "");
  const links = s
    ? `\n\n🔎 Screener: ${s}/\n📋 Follow-up segnali: ${s}/signals.html\n📖 Il metodo: ${s}/metodo.html`
    : "";
  return `👋 Benvenuto nello <b>Screener Weinstein</b>!\n`
    + `Segnala i candidati <i>long</i> secondo il metodo di Stan Weinstein su USA + Europa.`
    + links
    + `\n\nRiceverai il <b>recap settimanale</b> ogni sabato. Scrivi /stop per disiscriverti.`;
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { headers: CORS });
    const url = new URL(request.url);

    // --- WEBHOOK Telegram: /start iscrive + benvenuto, /stop disiscrive ---
    if (request.method === "POST" && url.pathname === "/telegram") {
      if (request.headers.get("X-Telegram-Bot-Api-Secret-Token") !== env.TG_WEBHOOK_SECRET)
        return json({ error: "unauthorized" }, 401);
      let u;
      try { u = await request.json(); } catch { return json({ ok: true }); }
      const m = u.message || u.edited_message;
      if (m && m.chat && typeof m.text === "string") {
        const chat = String(m.chat.id), text = m.text.trim();
        if (text.startsWith("/start")) {
          const name = str(`${m.chat.first_name || ""} ${m.chat.last_name || ""}`, 60).trim();
          await env.DB.prepare("INSERT OR REPLACE INTO subscribers (chat_id, name, added_at) VALUES (?,?,?)")
            .bind(chat, name, new Date().toISOString()).run();
          await tg(env, "sendMessage", { chat_id: chat, text: welcome(env), parse_mode: "HTML", disable_web_page_preview: true });
        } else if (text.startsWith("/stop")) {
          await env.DB.prepare("DELETE FROM subscribers WHERE chat_id=?").bind(chat).run();
          await tg(env, "sendMessage", { chat_id: chat, text: "Disiscritto dal recap settimanale. Scrivi /start per riattivarlo." });
        } else {
          await tg(env, "sendMessage", { chat_id: chat, text: welcome(env), parse_mode: "HTML", disable_web_page_preview: true });
        }
      }
      return json({ ok: true });   // Telegram vuole sempre 200
    }

    // --- letture per il JOB (WATCH_SECRET) ---
    if (request.method === "GET" && url.pathname === "/list") {
      if (url.searchParams.get("secret") !== env.WATCH_SECRET) return json({ error: "unauthorized" }, 401);
      const { results } = await env.DB.prepare(
        "SELECT ticker, signal_date, market, entry, stop, base_len, mansfield, vol_ratio, currency, added_at FROM follows ORDER BY added_at DESC"
      ).all();
      return json(results || []);
    }
    if (request.method === "GET" && url.pathname === "/subscribers") {
      if (url.searchParams.get("secret") !== env.WATCH_SECRET) return json({ error: "unauthorized" }, 401);
      const { results } = await env.DB.prepare("SELECT chat_id, name, added_at FROM subscribers").all();
      return json(results || []);
    }

    // --- scritture watchlist (WATCH_SECRET nel body) ---
    if (request.method === "POST") {
      let b;
      try { b = await request.json(); } catch { return json({ error: "bad json" }, 400); }
      if (b.secret !== env.WATCH_SECRET) return json({ error: "unauthorized" }, 401);
      const tk = str(b.ticker, 40), d = str(b.date, 10);
      if (!tk || !d) return json({ error: "missing ticker/date" }, 400);
      if (url.pathname === "/follow") {
        await env.DB.prepare(
          "INSERT OR REPLACE INTO follows (ticker, signal_date, market, entry, stop, base_len, mansfield, vol_ratio, currency, added_at) VALUES (?,?,?,?,?,?,?,?,?,?)"
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
