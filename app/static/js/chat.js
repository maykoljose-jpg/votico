const openBtn = document.getElementById('openChat');
const closeBtn = document.getElementById('closeChat');
const panel = document.getElementById('chatPanel');
const form = document.getElementById('chatForm');
const input = document.getElementById('chatInput');
const log = document.getElementById('chatLog');

openBtn.onclick = ()=>{ panel.hidden = false; input.focus(); };
closeBtn.onclick = ()=>{ panel.hidden = true; };

function addMsg(role, html){
  const div = document.createElement('div');
  div.className = `msg msg--${role}`;
  div.innerHTML = html;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const q = input.value.trim();
  if(!q) return;
  addMsg('user', q);
  input.value='';
  addMsg('bot', '<em>Buscando en los planes…</em>');

  try {
    const r = await fetch(`/api/chat`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({query:q})
    });
    const data = await r.json();
    log.lastChild.innerHTML = (data.answer || 'Sin respuesta').replace(/\n/g,'<br>');
    if (data.citations?.length){
      const cite = document.createElement('div');
      cite.className = 'citations';
      cite.innerHTML = '<strong>Fuentes:</strong><br>' + data.citations.slice(0,4).map(c =>
        `${c.party} — ${c.title} (p. ${c.page})`
      ).join('<br>');
      log.appendChild(cite);
      log.scrollTop = log.scrollHeight;
    }
  } catch(err){
    log.lastChild.innerHTML = 'Error contactando la API.';
  }
});
