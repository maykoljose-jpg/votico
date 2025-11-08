(function () {
  const elThread = document.getElementById("thread");
  const elForm = document.getElementById("ask-form");
  const elInput = document.getElementById("q-input");
  const elBtn = document.getElementById("ask-btn");

  // Un “id de sesión” simple por pestaña para agrupar conversación en localStorage
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

  // Helpers -------------
  const qs = new URLSearchParams(location.search);
  const initialQ = (qs.get("q") || "").trim();

  function saveThread() {
    // Guardar por sesión
    const state = JSON.parse(localStorage.getItem(SESSION_KEY) || "{}");
    state[sessionId] = elThread.innerHTML;
    localStorage.setItem(SESSION_KEY, JSON.stringify(state));
  }

  function restoreThread() {
    const state = JSON.parse(localStorage.getItem(SESSION_KEY) || "{}");
    if (state[sessionId]) {
      elThread.innerHTML = state[sessionId];
      requestAnimationFrame(scrollToBottom);
    }
  }

  function scrollToBottom() {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }

  function bubbleUser(text) {
    const div = document.createElement("div");
    div.className = "flex justify-end";
    div.innerHTML = `
      <div class="max-w-[90%] md:max-w-[80%] rounded-2xl px-4 py-3 bg-indigo-600 text-white shadow">
        <div class="text-sm opacity-80 mb-1">Tú</div>
        <div class="whitespace-pre-wrap leading-relaxed">${escapeHtml(text)}</div>
      </div>`;
    elThread.appendChild(div);
  }

  function bubbleAssistant(answer, citations) {
    const div = document.createElement("div");
    div.className = "flex justify-start";
    const citesHtml = (citations || [])
      .map((c) => {
        const party = (c.party || "desconocido");
        const title = (c.title || "");
        const page = c.page ? ` (p. ${c.page})` : "";
        return `<li>${escapeHtml(party)} — ${escapeHtml(title)}${page}</li>`;
      })
      .join("");
    div.innerHTML = `
      <div class="max-w-[90%] md:max-w-[80%] rounded-2xl px-4 py-4 bg-slate-800 text-slate-100 shadow space-y-4">
        <div class="text-sm opacity-80">Respuesta</div>
        <div class="prose prose-invert whitespace-pre-wrap leading-relaxed">${linkify(escapeHtml(answer))}</div>
        ${
          citesHtml
            ? `<div class="pt-2 border-t border-slate-700">
                 <div class="text-sm font-semibold mb-1">Fuentes:</div>
                 <ul class="list-disc pl-5 text-slate-300">${citesHtml}</ul>
               </div>`
            : ""
        }
      </div>`;
    elThread.appendChild(div);
  }

  function bubbleLoader() {
    const div = document.createElement("div");
    div.className = "flex justify-start";
    div.innerHTML = `
      <div class="max-w-[90%] md:max-w-[80%] rounded-2xl px-4 py-4 bg-slate-800 text-slate-300 shadow">
        Pensando…
      </div>`;
    elThread.appendChild(div);
    return div;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (m) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[m]));
  }

  function linkify(text) {
    // hace clicables URLs planas si llegaran a aparecer
    return text.replace(
      /(https?:\/\/[^\s)]+)([)\s]?)/g,
      '<a class="underline text-indigo-400 hover:text-indigo-300" target="_blank" rel="noopener noreferrer" href="$1">$1</a>$2'
    );
  }

  async function ask(query) {
    if (!query) return;
    bubbleUser(query);
    saveThread();
    scrollToBottom();

    const loader = bubbleLoader();
    scrollToBottom();

    try {
      const r = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await r.json();
      loader.remove();

      if (!r.ok) {
        bubbleAssistant(
          data?.detail || "No pude procesar la pregunta en este momento.",
          []
        );
      } else {
        bubbleAssistant(data.answer || "", data.citations || []);
      }
    } catch (e) {
      loader.remove();
      bubbleAssistant("Error de red. Intentalo de nuevo.", []);
    } finally {
      saveThread();
      requestAnimationFrame(scrollToBottom);
    }
  }

  // Eventos -------------
  elForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const q = elInput.value.trim();
    if (!q) return;
    elInput.value = "";
    ask(q);
  });

  // Restaurar conversación si la había
  restoreThread();

  // Si llegó ?q=... disparar la primera pregunta como mensaje del usuario
  if (initialQ) {
    // Evita duplicar si ya restauraste un hilo con ese mismo primer mensaje
    const alreadyHas = elThread.textContent.includes(initialQ);
    if (!alreadyHas) {
      ask(initialQ);
    }
  }

})();
