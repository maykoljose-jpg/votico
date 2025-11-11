# -*- coding: utf-8 -*-
"""
RAG conversacional y neutral por partido (con autodetección de proveedor de embeddings).

- Carga embeddings/metadata desde Cloudflare R2 o disco local.
- Índice cacheado en memoria (similitud coseno).
- Embeddings de consulta:
    * AUTO: se decide por la dimensión del índice (1536=OpenAI, 768=Gemini)
    * o fijo con EMBED_PROVIDER=openai|gemini
- El contexto se agrupa por partido para balance.
- Si no hay evidencia en planes: respuesta general (educativa) + sugerencia de portales Asamblea.

Responde con:
  - answer (texto)
  - citations (lista clásica)
  - sources_inline (una línea compacta)
  - sources_more (lista compacta para UI)

ENV:
  R2_PUBLIC_BASE, INDEX_PREFIX
  OPENAI_API_KEY, OPENAI_EMBED_MODEL=text-embedding-3-small
  OPENAI_MODEL=gpt-4o-mini
  GOOGLE_API_KEY, GEMINI_EMBED_MODEL=text-embedding-004
  TOP_K=6, ANSWER_TEMPERATURE=0.35, ANSWER_MAX_TOKENS=600
  MOCK_MODE="1" (demo)
  EMBED_PROVIDER="auto|openai|gemini" (auto por defecto)
"""

from __future__ import annotations
import os, json, logging
from io import BytesIO
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import httpx

# ====== SDKs opcionales ======
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except Exception:
    _GEMINI_AVAILABLE = False

# ====== Logging ======
logger = logging.getLogger("rag")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ---------------------------------------------------
# Utilidades descarga índice (Cloudflare R2 / local)
# ---------------------------------------------------
def _join(base: str, *parts: str) -> str:
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
    base = os.getenv("R2_PUBLIC_BASE", "").strip()
    if not base:
        raise RuntimeError("R2_PUBLIC_BASE no está definido.")
    prefix = os.getenv("INDEX_PREFIX", "index-3").strip().strip("/")
    meta_url = _join(base, prefix, "metadata.json")
    emb_url  = _join(base, prefix, "embeddings.npy")

    meta_bytes = _fetch_bytes(meta_url)
    emb_bytes  = _fetch_bytes(emb_url)

    metadata  = json.loads(meta_bytes.decode("utf-8"))
    embeddings = np.load(BytesIO(emb_bytes), allow_pickle=False)
    if not isinstance(embeddings, np.ndarray):
        raise RuntimeError("embeddings.npy no es un ndarray válido.")
    return embeddings, metadata

def _load_local_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    base_dir = os.path.join(os.path.dirname(__file__), "data", "index")
    emb_path = os.path.join(base_dir, "embeddings.npy")
    meta_path = os.path.join(base_dir, "metadata.json")
    logger.info(f"Usando índice local: {base_dir}")
    if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"No se encontró índice local en {base_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    embeddings = np.load(emb_path, allow_pickle=False)
    return embeddings, metadata

# -----------------
# Proveedor helpers
# -----------------
def _ensure_openai() -> OpenAI:
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("Paquete 'openai' no disponible.")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY no está definido.")
    return OpenAI(api_key=key)

def _ensure_gemini() -> None:
    if not _GEMINI_AVAILABLE:
        raise RuntimeError("Paquete 'google-generativeai' no disponible.")
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GOOGLE_API_KEY no está definido.")
    genai.configure(api_key=key)

def _pick_provider_by_dim(dim: int) -> str:
    """
    Si EMBED_PROVIDER=auto -> elegir por dimensión:
      1536 => openai, 768 => gemini
    Si EMBED_PROVIDER=openai|gemini -> respeta el override (y valida dimensión).
    """
    override = (os.getenv("EMBED_PROVIDER", "auto") or "auto").lower().strip()
    if override in ("openai", "gemini"):
        # Permitimos override; avisamos si dim no coincide (pero seguimos)
        expected = 1536 if override == "openai" else 768
        if dim != expected:
            logger.warning(f"[RAG] EMBED_PROVIDER={override} pero el índice es dim={dim} (esperado {expected}). "
                           f"Asegurate de reindexar con el mismo proveedor.")
        return override

    # auto
    if dim == 1536:
        return "openai"
    if dim == 768:
        return "gemini"
    raise RuntimeError(f"No sé qué proveedor usar para índice con dim={dim}. Usa EMBED_PROVIDER=openai|gemini.")

# -------------
#   RAG Index
# -------------
class RAGIndex:
    def __init__(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        if embeddings.ndim != 2:
            raise ValueError("Embeddings debe ser 2D (N, D).")
        if embeddings.shape[0] != len(metadata):
            raise ValueError("metadata y embeddings desalineados.")
        self._emb = embeddings.astype(np.float32, copy=False)
        norms = np.linalg.norm(self._emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._emb = self._emb / norms
        self._meta = metadata
        self.dim = self._emb.shape[1]
        self.provider = _pick_provider_by_dim(self.dim)
        logger.info(f"RAGIndex cargado: {self._emb.shape[0]} vectores, dim={self.dim}, provider={self.provider}")

    @property
    def size(self) -> int:
        return self._emb.shape[0]

    # --- búsqueda por vector ---
    def search_by_vector(self, query_vec: np.ndarray, top_k: int = 6) -> List[Dict[str, Any]]:
        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim != 1:
            raise ValueError("query_vec debe ser 1D")
        if q.shape[0] != self.dim:
            raise ValueError(f"Dim mismatch: query={q.shape[0]} vs index={self.dim}")
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = self._emb @ q
        k = max(1, min(int(top_k), self.size))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [{"score": float(sims[i]), "metadata": self._meta[i]} for i in idx]

    # --- embeddings de texto (elige proveedor según índice) ---
    def embed_text(self, text: str) -> np.ndarray:
        if self.provider == "gemini":
            _ensure_gemini()
            model = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
            resp = genai.embed_content(model=model, content=text, task_type="RETRIEVAL_QUERY")
            vec = np.asarray(resp["embedding"], dtype=np.float32)
            if vec.shape[0] != self.dim:
                raise RuntimeError(f"Embeddings Gemini dim={vec.shape[0]} no coincide con índice dim={self.dim}.")
            return vec

        # openai
        client = _ensure_openai()
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        resp = client.embeddings.create(model=model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        if vec.shape[0] != self.dim:
            raise RuntimeError(f"Embeddings OpenAI dim={vec.shape[0]} no coincide con índice dim={self.dim}.")
        return vec

    def search_by_text(self, text: str, top_k: int = 6) -> List[Dict[str, Any]]:
        vec = self.embed_text(text)
        return self.search_by_vector(vec, top_k=top_k)

# ---------------
# Carga en caché
# ---------------
_INDEX: Optional[RAGIndex] = None

def get_index(force_reload: bool = False) -> RAGIndex:
    global _INDEX
    if _INDEX is not None and not force_reload:
        return _INDEX
    try:
        emb, meta = _load_remote_index()
        logger.info("Índice cargado desde R2.")
    except Exception as e:
        logger.warning(f"No se pudo cargar desde R2: {e}. Probando local…")
        emb, meta = _load_local_index()
        logger.info("Índice cargado desde disco local.")
    _INDEX = RAGIndex(embeddings=emb, metadata=meta)
    return _INDEX

# -------------
# Facades
# -------------
def retrieve_by_text(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    return get_index().search_by_text(query, top_k=top_k)

def retrieve_by_vector(query_vec: List[float], top_k: int = 6) -> List[Dict[str, Any]]:
    v = np.asarray(query_vec, dtype=np.float32)
    return get_index().search_by_vector(v, top_k=top_k)

def top_k_context(query: str, top_k: int = 6, key_field: str = "text") -> List[str]:
    hits = retrieve_by_text(query, top_k=top_k)
    out = []
    for h in hits:
        md = h["metadata"] or {}
        out.append(md.get(key_field) or md.get("chunk") or md.get("content") or "")
    return out

# -----------------------
# Formato de contexto y citas
# -----------------------
def _format_sources_grouped(hits: List[Dict[str, Any]], per_party: int = 2) -> str:
    """
    Bloque de contexto agrupado por partido (máx per_party por partido).
    """
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

def _citations_compact(hits: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Devuelve (sources_inline, sources_more) en formato compacto.
    """
    items = []
    inline_parts = []
    for i, h in enumerate(hits, start=1):
        md = h.get("metadata", {}) or {}
        party = (md.get("party") or "desconocido").strip()
        title = (md.get("title") or "").strip()
        page  = md.get("page", "")
        items.append(f"[{i}] {party} — {title} (p. {page})")
        short = f"{party} (p. {page})" if page != "" else party
        inline_parts.append(short)

    seen = set()
    inline = []
    for p in inline_parts:
        if p not in seen:
            seen.add(p)
            inline.append(p)
    inline_str = ", ".join(inline[:8]) + ("…" if len(inline) > 8 else "")
    return inline_str, items

# ----------------
# Mensajes al LLM
# ----------------
def _build_messages(query: str, hits: List[Dict[str, Any]],
                    history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    context_block = _format_sources_grouped(hits, per_party=2)
    guidance = (
        "Sos un asistente neutral que compara propuestas de partidos en Costa Rica.\n"
        "Usá SOLO la evidencia del contexto provisto. Si un partido no aparece, indicá que no se observa propuesta en los fragmentos.\n"
        "Redactá en tono natural, claro y amable. Evitá muletillas y cierres repetidos.\n"
        "Estructura sugerida (adaptala si hace falta):\n"
        "• Mini-resumen por partido (2–3 oraciones por partido) con citas [n] pegadas a las frases.\n"
        "• Una comparación breve (2–4 oraciones) con [n] si aplica.\n"
        "• Cerrá con UNA pregunta útil y específica (comparar dos partidos, pedir enfoque, región, etc.)."
    )
    msgs = [{"role": "system", "content": guidance}]

    if history:
        for m in history[-10:]:
            r = m.get("role")
            c = m.get("content", "")
            if r in ("user", "assistant") and c:
                msgs.append({"role": r, "content": c})

    user = (
        f"Consulta: {query}\n\n"
        f"Contexto (fragmentos agrupados por partido):\n{context_block}\n\n"
        "Redactá la respuesta siguiendo el estilo indicado y con citas [n] precisas."
    )
    msgs.append({"role": "user", "content": user})
    return msgs

async def _openai_chat(messages, model: str, temperature: float, max_tokens: int, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

# ----------------------------------
# Fallback: explicación general
# ----------------------------------
async def _general_explain(query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return (
            "No encontré referencias directas a ese tema en los planes de gobierno. "
            "Si te sirve, puedo explicarlo a nivel general y cómo suele relacionarse con políticas públicas en Costa Rica. "
            "También, es posible buscar información en los portales de la Asamblea Legislativa: "
            "‘Consultas SIL’ o ‘Consulta de Proyectos’. ¿Querés que te oriente?"
        )

    msgs: List[Dict[str, str]] = [
        {"role": "system",
         "content": (
             "Sos un asistente neutral para Costa Rica. Cuando no hay evidencia en planes de gobierno, "
             "dá una explicación general (6–8 oraciones) de qué es, por qué importa y cómo suele abordarse en política pública en CR. "
             "No infieras posturas de partidos. Cerrá con una pregunta concreta para avanzar."
         )}]
    if history:
        for t in history[-6:]:
            role = t.get("role", "user")
            content = t.get("content", "")
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})

    msgs.append({"role": "user",
                 "content": (
                     f"Tema del usuario: {query}\n"
                     "Dá la explicación general. Podés sugerir revisar los sitios de la Asamblea Legislativa: "
                     "https://www.asamblea.go.cr/Centro_de_informacion/Consultas_SIL/SitePages/SIL.aspx y "
                     "https://www.asamblea.go.cr/Centro_de_informacion/Consultas_SIL/SitePages/ConsultaProyectos.aspx"
                 )})

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temp = float(os.getenv("ANSWER_TEMPERATURE", "0.35"))
    try:
        text = await _openai_chat(msgs, model, temp, 380, api_key)
        return text.strip()
    except Exception:
        return (
            "No vi referencias al tema en los planes de gobierno. "
            "Si querés, te explico el concepto en pocas palabras y cómo suele abordarse en CR. "
            "También podemos revisar los portales de la Asamblea Legislativa que acabo de mencionar. ¿Cómo preferís seguir?"
        )

# -------------
#   answer()
# -------------
async def answer(query: str, top_k: int = None, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    k = int(top_k or os.getenv("TOP_K", "6"))
    hits = retrieve_by_text(query, top_k=k)

    if not hits:
        general = await _general_explain(query, history=history)
        return {
            "answer": general,
            "citations": [],
            "sources_inline": "",
            "sources_more": []
        }

    sources_inline, sources_more = _citations_compact(hits)

    citations = []
    for h in hits:
        md = h.get("metadata", {}) or {}
        citations.append({
            "party":  str(md.get("party", "")),
            "title":  str(md.get("title", "")),
            "page":   md.get("page", ""),
            "source": str(md.get("source", "")),
            "score":  float(h.get("score", 0.0)),
        })

    if os.getenv("MOCK_MODE", "0") == "1":
        return {
            "answer": (
                "Te dejo un panorama por partido basado en los fragmentos más cercanos [1][2][3]. "
                "Si querés, comparo dos partidos o profundizo en un punto."
            ),
            "citations": citations,
            "sources_inline": sources_inline,
            "sources_more": sources_more
        }

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        bullets = []
        for i, h in enumerate(hits, 1):
            md = h.get("metadata", {}) or {}
            t = (md.get("chunk") or md.get("text") or "")[:220].replace("\n"," ")
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
    msgs = _build_messages(query, hits, history=history)

    try:
        text = await _openai_chat(msgs, model, temperature, max_tokens, api_key)
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
            t = (md.get("chunk") or md.get("text") or "")[:220].replace("\n"," ")
            bullets.append(f"[{i}] {t}")
        return {
            "answer": "No pude conectarme al modelo en este momento, pero sí hay material en los planes. "
                      "Resumen rápido:\n- " + "\n- ".join(bullets),
            "citations": citations,
            "sources_inline": sources_inline,
            "sources_more": sources_more
        }

# -----------------------
#   Salud / estadísticas
# -----------------------
async def check_openai_connectivity() -> Dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not key:
        return {"ok": False, "error": "Falta OPENAI_API_KEY"}
    try:
        msg = [{"role": "user", "content": "Di 'ping'."}]
        txt = await _openai_chat(msg, model, 0.0, 5, key)
        return {"ok": True, "reply": txt[:50], "status": 200, "message": "Conexión OK y API key válida"}
    except httpx.TimeoutException:
        return {"ok": False, "error": "Timeout a OpenAI"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def index_stats() -> Dict[str, Any]:
    idx = get_index()
    parties = Counter([(m.get("party") or "desconocido") for m in getattr(idx, "_meta", [])])
    return {"vectors": idx.size, "dim": idx.dim, "parties": dict(parties)}
