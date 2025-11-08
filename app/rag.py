# -*- coding: utf-8 -*-
"""
RAG minimalista + respuesta conversacional y neutral por partido.

- Carga embeddings + metadata desde Cloudflare R2 (público) o disco local.
- Índice cacheado en memoria (normalizado) para similitud coseno.
- Consulta por texto usando OpenAI embeddings (dim 1536) o por vector.
- Balancea los fragmentos por partido para evitar sesgos.
- Respuesta con tono natural y cierre dinámico (sin plantilla fija).
- Fallback educativo si no hay evidencia y enlaces a Asamblea Legislativa.

ENV esperadas:
  R2_PUBLIC_BASE            p.ej. https://pub-xxxxxxxxxxxxxxx.r2.dev
  INDEX_PREFIX              p.ej. index-3
  OPENAI_API_KEY
  OPENAI_EMBED_MODEL        default: text-embedding-3-small (1536)
  OPENAI_MODEL              default: gpt-4o-mini
  TOP_K                     default: 6
  ANSWER_TEMPERATURE        default: 0.35
  ANSWER_MAX_TOKENS         default: 600
  MOCK_MODE                 "1" para demo (no llama a OpenAI)
"""

import os, json, logging
from io import BytesIO
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import httpx

# OpenAI SDK (embeddings + chat)
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    _OPENAI_AVAILABLE = False

# ---- logging básico
logger = logging.getLogger("rag")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# =========================
#   utilidades de descarga
# =========================

def _join_public_url(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    tail = "/".join(p.strip("/") for p in parts if p and p != "/")
    return f"{base}/{tail}"

def _fetch_bytes(url: str, timeout: int = 30) -> bytes:
    logger.info(f"Descargando: {url}")
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.content

def _load_remote_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Descarga embeddings.npy y metadata.json desde R2 público."""
    base = os.getenv("R2_PUBLIC_BASE", "").strip()
    if not base:
        raise RuntimeError("R2_PUBLIC_BASE no está definido.")
    prefix = os.getenv("INDEX_PREFIX", "index-3").strip().strip("/")
    meta_url = _join_public_url(base, prefix, "metadata.json")
    emb_url  = _join_public_url(base, prefix, "embeddings.npy")
    metadata_bytes = _fetch_bytes(meta_url)
    embeddings_bytes = _fetch_bytes(emb_url)
    metadata  = json.loads(metadata_bytes.decode("utf-8"))
    embeddings = np.load(BytesIO(embeddings_bytes), allow_pickle=False)
    if not isinstance(embeddings, np.ndarray):
        raise RuntimeError("embeddings.npy no es un ndarray válido.")
    return embeddings, metadata

def _load_local_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Fallback local: app/data/index/{embeddings.npy, metadata.json}"""
    base_dir = os.path.join(os.path.dirname(__file__), "data", "index")
    emb_path = os.path.join(base_dir, "embeddings.npy")
    meta_path = os.path.join(base_dir, "metadata.json")
    logger.info(f"Usando índice local: {base_dir}")
    if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"No se encontró el índice local en {base_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    embeddings = np.load(emb_path, allow_pickle=False)
    return embeddings, metadata

# =========================
#          índice
# =========================

class RAGIndex:
    """Índice en memoria con embeddings normalizados (coseno)."""

    def __init__(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        if embeddings.ndim != 2:
            raise ValueError("Embeddings deben ser (N, D).")
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(f"metadata ({len(metadata)}) != embeddings ({embeddings.shape[0]}).")

        self._emb = embeddings.astype(np.float32, copy=False)
        norms = np.linalg.norm(self._emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._emb = self._emb / norms

        self._meta = metadata
        self.dim = self._emb.shape[1]
        logger.info(f"RAGIndex cargado: {self._emb.shape[0]} vectores, dim={self.dim}")

    @property
    def size(self) -> int:
        return self._emb.shape[0]

    def search_by_vector(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim != 1:
            raise ValueError("query_vec debe ser 1D (dim,).")
        if q.shape[0] != self.dim:
            raise ValueError(f"Dim mismatch: query={q.shape[0]} vs index={self.dim}")
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        sims = self._emb @ q_norm  # (N,)
        top_k = max(1, min(int(top_k), self.size))
        idx = np.argpartition(-sims, top_k - 1)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        return [{"score": float(sims[i]), "metadata": self._meta[i]} for i in idx]

    # --- OpenAI helpers
    def _ensure_openai(self):
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("Paquete 'openai' no disponible.")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no está definido.")
        return OpenAI(api_key=api_key)

    def embed_text(self, text: str) -> np.ndarray:
        client = self._ensure_openai()
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        resp = client.embeddings.create(model=model, input=text)
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def search_by_text(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        vec = self.embed_text(text)
        return self.search_by_vector(vec, top_k=top_k)

# =========================
#   carga / caché de índice
# =========================

_INDEX: Optional[RAGIndex] = None

def get_index(force_reload: bool = False) -> RAGIndex:
    global _INDEX
    if _INDEX is not None and not force_reload:
        return _INDEX
    try:
        embeddings, metadata = _load_remote_index()
        logger.info("Índice cargado desde Cloudflare R2.")
    except Exception as e_remote:
        logger.warning(f"No se pudo cargar desde R2: {e_remote}. Probando local…")
        embeddings, metadata = _load_local_index()
        logger.info("Índice cargado desde disco local.")
    _INDEX = RAGIndex(embeddings=embeddings, metadata=metadata)
    return _INDEX

# =========================
#      helpers de RAG
# =========================

def _rebalance_hits(hits: List[Dict[str, Any]], per_party: int = 2, max_total: int = 8) -> List[Dict[str, Any]]:
    """Intercala hits por partido para evitar dominancia de uno solo."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    for h in hits:
        md = h.get("metadata", {}) or {}
        party = md.get("party") or "Desconocido"
        if party not in buckets:
            order.append(party)
        if len(buckets[party]) < per_party:
            buckets[party].append(h)

    merged, layer = [], 0
    while len(merged) < max_total:
        advanced = False
        for p in order:
            if layer < len(buckets[p]):
                merged.append(buckets[p][layer])
                if len(merged) >= max_total:
                    break
                advanced = True
        if not advanced:
            break
        layer += 1
    return merged

def _format_sources_grouped(hits: List[Dict[str, Any]], per_party: int = 2) -> str:
    """Arma el bloque de contexto agrupando por partido (máx. `per_party` por partido)."""
    grouped = defaultdict(list)
    for i, h in enumerate(hits, start=1):
        md = h.get("metadata", {}) or {}
        party = md.get("party") or "Desconocido"
        title = md.get("title", "")
        page  = md.get("page", "")
        src   = md.get("source", "")
        chunk = (md.get("chunk") or md.get("text") or md.get("content") or "").strip()
        snippet = chunk[:600].replace("\n", " ")
        grouped[party].append(f"[{i}] {title} – pág. {page} – {src}\n>>> {snippet}")
    blocks = []
    for party, entries in grouped.items():
        blocks.append(f"### {party}\n" + "\n\n".join(entries[:per_party]))
    return "\n\n".join(blocks)

def _compact_sources(cites: List[Dict[str, Any]], limit: int = 6) -> Tuple[List[str], int]:
    """Resumen de fuentes para UI: hasta `limit` ítems + contador 'más'."""
    items = []
    for c in cites[:limit]:
        party = c.get("party",""); title = c.get("title",""); page = c.get("page","")
        items.append(f"{party} — {title} (p. {page})")
    more = max(0, len(cites) - limit)
    return items, more

# =========================
#        mensajes LLM
# =========================

def _build_messages(
    query: str,
    hits: List[Dict[str, Any]],
    style: str = "CONVERSATIONAL",
    history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:

    context_block = _format_sources_grouped(hits, per_party=2)

    guidance = (
        "Sos un asistente neutral que compara propuestas en Costa Rica.\n"
        "Usá SOLO el contexto provisto (fragmentos de planes). Si un partido no aparece, indicá que no se observa propuesta en el contexto.\n"
        "Cubrí a cada partido presente de forma balanceada (2–3 oraciones por partido), luego hacé una comparación breve (2–4 oraciones).\n"
        "Tono natural, claro, sin jerga densa, en ~10–14 oraciones totales. Pegá las citas [n] en el lugar exacto.\n"
        "Cerrá con UNA pregunta breve **dinámica** adecuada al tema (nada de plantillas fijas)."
    )

    msgs: List[Dict[str, str]] = [{"role": "system", "content": guidance}]

    if history:
        for m in history[-10:]:
            r = m.get("role"); c = m.get("content", "")
            if r in ("user","assistant") and c:
                msgs.append({"role": r, "content": c})

    user_prompt = (
        f"Pregunta del usuario:\n{query}\n\n"
        f"Contexto agrupado por partido (fragmentos de planes de gobierno):\n{context_block}\n\n"
        "Tarea: redactá la respuesta siguiendo la estructura indicada, con citas [n] precisas y cierre con UNA pregunta pertinente."
    )
    msgs.append({"role":"user","content":user_prompt})
    return msgs

# =========================
#   llamada a OpenAI chat
# =========================

async def _openai_chat(messages, model: str, temperature: float, max_tokens: int, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

# =========================
# fallback si no hay matches
# =========================

async def _general_explain(query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Explicación neutral cuando no hay evidencia en los planes.
    Ofrece buscar en Asamblea Legislativa (enlaces), sin scrapear automáticamente.
    """
    base_msg = (
        "En los fragmentos de planes de gobierno revisados no vi propuestas directas sobre este tema. "
        "Si querés, puedo explicarlo a nivel general en el contexto de Costa Rica y, además, "
        "podemos consultar fuentes legislativas. Por ejemplo:\n"
        "• Búsqueda de proyectos y antecedentes (SIL): "
        "https://www.asamblea.go.cr/Centro_de_informacion/Consultas_SIL/SitePages/SIL.aspx\n"
        "• Consulta directa de proyectos: "
        "https://www.asamblea.go.cr/Centro_de_informacion/Consultas_SIL/SitePages/ConsultaProyectos.aspx\n\n"
        "¿Preferís una explicación breve o que te oriente para buscar ahí?"
    )

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return base_msg

    msgs: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Sos un asistente neutral para Costa Rica. Cuando el tema no aparece en los planes, "
                "dá una explicación general (qué es, por qué importa, cómo suele abordarse en políticas públicas), "
                "sin inferir posturas de partidos. Tono claro y breve (6–9 oraciones). "
                "Cerrá ofreciendo la opción de buscar en los sitios legislativos enlazados y con una sola pregunta final."
            ),
        }
    ]
    if history:
        for t in history[-6:]:
            if t.get("role") in ("user","assistant") and t.get("content"):
                msgs.append({"role": t["role"], "content": t["content"]})

    msgs.append({
        "role": "user",
        "content": f"Tema del usuario: {query}\nExplicalo en general y ofrecé buscar en las fuentes legislativas enlazadas."
    })

    try:
        text = await _openai_chat(
            msgs,
            os.getenv("OPENAI_MODEL","gpt-4o-mini"),
            float(os.getenv("ANSWER_TEMPERATURE","0.35")),
            300,
            api_key
        )
        return text.strip() + "\n\n" + base_msg
    except Exception:
        return base_msg

# =========================
#          answer()
# =========================

async def answer(query: str, top_k: int = None, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    k = int(top_k or os.getenv("TOP_K", "6"))

    # traemos más de lo que mostraremos para poder balancear por partido
    hits_raw = get_index().search_by_text(query, top_k=k * 3)
    hits = _rebalance_hits(hits_raw, per_party=2, max_total=k)

    if not hits:
        general = await _general_explain(query, history=history)
        return {
            "answer": (
                "En los planes de gobierno que revisamos no veo propuestas directas sobre este tema. "
                + general
            ),
            "citations": []
        }

    # citas completas
    citations: List[Dict[str, Any]] = []
    for h in hits:
        md = h.get("metadata", {}) or {}
        citations.append({
            "party":  str(md.get("party", "")),
            "title":  str(md.get("title", "")),
            "page":   md.get("page", ""),
            "source": str(md.get("source", "")),
            "score":  float(h.get("score", 0.0)),
        })

    # compactar para UI
    sources_inline, sources_more = _compact_sources(citations, limit=6)

    # demo
    if os.getenv("MOCK_MODE", "0") == "1":
        return {
            "answer": "Resumen comparativo breve basado en fragmentos agrupados por partido [1][2][3]. ¿Querés profundizar en alguno?",
            "citations": citations,
            "sources_inline": sources_inline,
            "sources_more": sources_more
        }

    # sin API key -> resumen mínimo
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        bullets = []
        for i, h in enumerate(hits, 1):
            md = h.get("metadata", {}) or {}
            t = (md.get("chunk") or md.get("text") or "")[:200].replace("\n"," ")
            bullets.append(f"[{i}] {t}")
        return {
            "answer": "Encontré fragmentos relevantes, pero no pude llamar al modelo (falta API key). "
                      "Resumen mínimo:\n- " + "\n- ".join(bullets),
            "citations": citations,
            "sources_inline": sources_inline,
            "sources_more": sources_more
        }

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("ANSWER_TEMPERATURE", "0.35"))
    max_tokens = int(os.getenv("ANSWER_MAX_TOKENS", "600"))
    messages = _build_messages(query, hits, style=os.getenv("ANSWER_STYLE","CONVERSATIONAL"), history=history)

    try:
        text = await _openai_chat(messages, model, temperature, max_tokens, api_key)
        return {
            "answer": text,
            "citations": citations,
            "sources_inline": sources_inline,
            "sources_more": sources_more
        }
    except Exception as e:
        logger.warning(f"Fallo al llamar OpenAI: {e}")
        bullets = []
        for i, h in enumerate(hits, 1):
            md = h.get("metadata", {}) or {}
            t = (md.get("chunk") or md.get("text") or "")[:200].replace("\n"," ")
            bullets.append(f"[{i}] {t}")
        return {
            "answer": "No pude conectarme al modelo en este momento, pero sí hay material en los planes. "
                      "Resumen rápido:\n- " + "\n- ".join(bullets),
            "citations": citations,
            "sources_inline": sources_inline,
            "sources_more": sources_more
        }

# =========================
# diagnósticos / utilidades
# =========================

async def check_openai_connectivity() -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return {"ok": False, "error": "Falta OPENAI_API_KEY"}
    try:
        msg = [{"role": "user", "content": "Di 'ping'."}]
        txt = await _openai_chat(msg, model, 0.0, 5, api_key)
        return {"ok": True, "reply": txt[:50], "status": 200, "message": "Conexión OK y API key válida"}
    except httpx.TimeoutException:
        return {"ok": False, "error": "Timeout a OpenAI"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def index_stats() -> Dict[str, Any]:
    idx = get_index()
    parties = Counter([(m.get("party") or "desconocido") for m in getattr(idx, "_meta", [])])
    return {"vectors": idx.size, "dim": idx.dim, "parties": dict(parties)}
