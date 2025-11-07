// app/static/js/chat.js
(function () {
  function start() {
    const elThread = document.getElementById("thread");
    const elForm = document.getElementById("ask-form");
    const elInput = document.getElementById("q-input");
    const elBtn = document.getElementById("ask-btn");

    // Defensa: si algo no existe, no seguimos.
    if (!elThread || !elForm || !elInput || !elBtn) {
      console.error("[chat] Faltan nodos del DOM", { elThread, elForm, elInput, elBtn });
      return;
    }

    const SESSION_KEY = "votico-thread";
    const sessionId = (() => {
      const k = "votico-session-id";
      let v = sessionStorage.getItem(k);
      if (!v) {
        v = "s" + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
        sessionStorage.setItem(k, v);
      }
      return v;
    })();

    const qs = new URLSearchParams(location.search);
    const initialQ = (qs.get("q") || "").trim();

    function saveThread() {
      try {
        const state = JSON.parse(localStorage.getItem(SESSION_KEY) || "{}");
        state[sessionId] = elThread.innerHTML;
        localStorage.setItem(SESSION_KEY, JSON.stringify(state));
      } catch (e) { console.warn("[chat] saveThread", e); }
    }
    function restoreThread() {
      try {
        const state = JSON.parse(localStorage.getItem(SESSION_KEY) || "{}");
        if (state[sessionId]) {
          elThread.innerHTML = state[sessionId];
          requestAnimationFrame(scrollToBottom);
        }
      } catch (e) { console.warn("[chat] restoreThread", e); }
    }
    function scrollToBottom() {
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    }
    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, (m) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m])
      );
    }
    function linkify(text) {
      return text.replace(
        /(https?:\/\/[^\s)]+)([)\s]?)/g,
        '<a target="_blank" rel="noopener noreferrer" href="$1">$1</a>$2'
      );
    }

    function bubble(kind, html) {
      const row = document.createElement("div");
      row.className = "row " + kind; // user | bot
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.innerHTML = html;
      row.appendChild(bubble);
      elThread.appendChild(row);
      return row;
    }
    function bubbleUser(text) {
      return bubble("user", escapeHtml(text));
    }
    function bubbleAssistant(answer, citations) {
      const citesHtml = (citations || [])
        .map((c) => {
          const party = c.party || "desconocido";
          const title = c.title || "";
          const page = c.page ? ` (p. ${c.page})` : "";
          return `<li>${escapeHtml(party)} — ${escapeHtml(title)}${page}</li>`;
        })
        .join("");
      const inner =
        `<div>${linkify(escapeHtml(answer || ""))}</div>` +
        (citesHtml ? `<div class="meta"><strong>Fuentes:</strong><ul>${citesHtml}</ul></div>` : "");
      return bubble("bot", inner);
    }
    function bubbleLoader() {
      return bubble("bot", `<span style="opacity:.85">Pensando…</span>`);
    }

    async function ask(query) {
      if (!query) return;
      try {
        bubbleUser(query);
        saveThread();
        scrollToBottom();

        const loader = bubbleLoader();
        scrollToBottom();

        const r = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query }),
        });

        let data = null;
        try { data = await r.json(); } catch {}
        loader.remove();

        if (!r.ok) {
          const msg = (data && (data.detail || data.error)) || "No pude procesar la pregunta.";
          bubbleAssistant(msg, []);
        } else {
          bubbleAssistant(data?.answer || "(sin respuesta)", data?.citations || []);
        }
      } catch (e) {
        console.error("[chat] ask error", e);
        bubbleAssistant("Error de red. Intentalo de nuevo.", []);
      } finally {
        saveThread();
        requestAnimationFrame(scrollToBottom);
      }
    }

    // Manejo de submit
    elForm.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const q = elInput.value.trim();
      if (!q) return;
      elInput.value = "";
      ask(q);
    });

    restoreThread();

    // Si viene ?q= en la URL y no está ya en el hilo, la disparamos
    if (initialQ) {
      const already = elThread.textContent.includes(initialQ);
      if (!already) ask(initialQ);
    }

    console.log("[chat] listo ✅");
  }

  // Esperar a que el DOM esté listo (evita que el form se envíe antes de atar eventos)
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
