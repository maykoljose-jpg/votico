// app/static/js/chat.js

// ===== Util: session id persistente en el navegador =====
function getSessionId() {
  try {
    let sid = localStorage.getItem("votico_session_id");
    if (!sid) {
      sid = crypto.randomUUID();
      localStorage.setItem("votico_session_id", sid);
    }
    return sid;
  } catch (_) {
    // fallback
    return Math.random().toString(36).slice(2);
  }
}

// ===== DOM refs =====
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const askBtn = document.getElementById("ask-btn");
const answerBox = document.getElementById("answer-box");
const sourcesList = document.getElementById("sources-list");

function setLoading(state) {
  if (state) {
    askBtn.disabled = true;
    askBtn.dataset.original = askBtn.textContent;
    askBtn.textContent = "Pensando…";
  } else {
    askBtn.disabled = false;
    askBtn.textContent = askBtn.dataset.original || "Preguntar";
  }
}

function renderAnswer(text) {
  if (!answerBox) return;
  answerBox.textContent = "";
  const p = document.createElement("p");
  p.textContent = text;
  answerBox.appendChild(p);
}

function renderSources(citations) {
  if (!sourcesList) return;
  sourcesList.innerHTML = "";
  (citations || []).forEach((c) => {
    const li = document.createElement("li");
    const party = c.party || "desconocido";
    const title = c.title || "—";
    const page = c.page !== undefined && c.page !== null ? ` (p. ${c.page})` : "";
    li.textContent = `${party} — ${title}${page}`;
    sourcesList.appendChild(li);
  });
}

// Submit por botón o Enter
if (form) {
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const q = (input.value || "").trim();
    if (!q) return;
    await doAsk(q);
  });
}

async function doAsk(question) {
  try {
    setLoading(true);
    renderAnswer("…");
    sourcesList.innerHTML = "";

    const sid = getSessionId();
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: question,
        session_id: sid,
        use_memory: true
      }),
    });

    const data = await resp.json();

    if (!resp.ok) {
      renderAnswer(data?.detail || "Hubo un error en el servidor.");
      return;
    }

    renderAnswer(data.answer || "Sin respuesta");
    renderSources(data.citations || []);
  } catch (e) {
    console.error(e);
    renderAnswer("No se pudo conectar. Intentá de nuevo.");
  } finally {
    setLoading(false);
    input.focus();
  }
}
// app/static/js/chat.js
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatContainer = document.getElementById("chat-container");

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = chatInput.value.trim();
  if (!query) return;

  // Agregar pregunta del usuario
  appendMessage("user", query);
  chatInput.value = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    const data = await res.json();
    appendMessage("bot", data.answer || "No se encontró respuesta.");
  } catch (err) {
    appendMessage("bot", "⚠️ Error al conectar con el servidor.");
  }
});

function appendMessage(sender, text) {
  const msg = document.createElement("div");
  msg.classList.add("msg", sender);
  msg.textContent = text;
  chatContainer.appendChild(msg);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}
