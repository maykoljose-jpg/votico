// static/js/chat.js
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatLog = document.getElementById("chatLog");

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;

  const userMsg = document.createElement("div");
  userMsg.className = "msg msg--user";
  userMsg.textContent = question;
  chatLog.appendChild(userMsg);

  chatInput.value = "";
  chatInput.disabled = true;

  const payload = { q: question };
  const loading = document.createElement("div");
  loading.className = "msg msg--bot";
  loading.textContent = "Pensando…";
  chatLog.appendChild(loading);
  chatLog.scrollTop = chatLog.scrollHeight;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    chatLog.removeChild(loading);

    const answerEl = document.createElement("div");
    answerEl.className = "msg msg--bot";
    answerEl.innerHTML = data.answer;
    chatLog.appendChild(answerEl);

    // === NUEVO: Mostrar fuentes compactas ===
    if (data.sources_inline || (data.sources_more && data.sources_more.length)) {
      const srcWrap = document.createElement("div");
      srcWrap.className = "citations";

      // Línea compacta
      if (data.sources_inline) {
        const p = document.createElement("p");
        p.style.margin = "6px 0";
        p.textContent = `Fuentes: ${data.sources_inline}`;
        srcWrap.appendChild(p);
      }

      // Lista detallada desplegable
      if (data.sources_more && data.sources_more.length) {
        const details = document.createElement("details");
        const sum = document.createElement("summary");
        sum.textContent = "Ver más";
        details.appendChild(sum);

        const ul = document.createElement("ul");
        ul.style.margin = "6px 0 0 16px";
        ul.style.padding = "0";

        data.sources_more.slice(0, 10).forEach(line => {
          const li = document.createElement("li");
          li.textContent = line;
          ul.appendChild(li);
        });

        details.appendChild(ul);
        srcWrap.appendChild(details);
      }

      answerEl.appendChild(srcWrap);
    }
    // === FIN NUEVO ===

    chatLog.scrollTop = chatLog.scrollHeight;
  } catch (err) {
    console.error(err);
    chatLog.removeChild(loading);

    const errorMsg = document.createElement("div");
    errorMsg.className = "msg msg--bot";
    errorMsg.textContent = "Ocurrió un error al conectar con el servidor.";
    chatLog.appendChild(errorMsg);
  } finally {
    chatInput.disabled = false;
    chatInput.focus();
    chatLog.scrollTop = chatLog.scrollHeight;
  }
});
