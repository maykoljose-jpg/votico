(function(){
  const btn = document.getElementById('btn-openai-check');
  const badge = document.getElementById('badge-openai');
  if(!btn || !badge) return;

  async function runCheck() {
    try {
      badge.textContent = 'Probando...';
      badge.className = 'badge badge--idle';
      const r = await fetch('/api/openai-check', { method: 'GET' });
      const data = await r.json();
      if (data.ok === true && data.status === 200) {
        badge.textContent = 'OpenAI: OK';
        badge.className = 'badge badge--ok';
      } else {
        badge.textContent = `OpenAI: ${data.status || 'error'}`;
        badge.className = 'badge badge--err';
      }
    } catch (e) {
      badge.textContent = 'OpenAI: error de red';
      badge.className = 'badge badge--err';
    }
  }

  btn.addEventListener('click', runCheck);
  window.addEventListener('load', runCheck);
})();
