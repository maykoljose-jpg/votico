# -*- coding: utf-8 -*-
"""
RAG conversacional y neutral por partido.

- Carga embeddings/metadata desde Cloudflare R2 o disco local.
- Índice cacheado en memoria (similitud coseno).
- Usa SIEMPRE el proveedor de embeddings consistente con la dimensión del índice:
    * dim=768  -> Gemini (text-embedding-004)
    * dim=1536 -> OpenAI (text-embedding-3-small)

- El contexto se agrupa por partido para balance.
- Si no hay evidencia en planes: respuesta general + sugerencia de portales Asamblea.
"""

from __future__ import annotations

import os
import json
import logging
from io import BytesIO
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import httpx

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
    prefix = os.getenv("INDEX_PREFIX", "index-gemini").strip().strip("/")
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

# -------------
#   RAG Index
# -------------
class RAGIndex:
    """Índice en memoria con coseno + autodetección de proveedor por dimensión."""
    def __init__(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        if embeddings.ndim != 2:
            raise ValueError("Embeddings deben ser 2D (N, D).")
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(f"metadata ({len(metadata)}) != embeddings ({embeddings.shape[0]})")

        self._emb = embeddings.astype(np.float32, copy=False)
        norms = np.linalg.norm(self._emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._emb = self._emb / norms

        self._meta = metadata
        self.dim = self._emb.shape[1]

        # Detectar proveedor según dimensión del índice
        if self.dim == 768:
            self.provider = "gemini"
        elif self.dim == 1536:
            self.provider = "openai"
        else:
            raise ValueError(f"Dim de índice no soportada: {self.dim}")

        logger.info(f"[RAG] Index: n={self._emb.shape[0]} dim={self.dim} provider={self.provider}")

    @property
    def size(self) -> int:
        return self._emb.shape[0]

    # ---------- Embeddings ----------
    def _embed_gemini(self, text: str) -> np.ndarray:
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError("Falta GOOGLE_API_KEY")
        genai.configure(api_key=api_key)

        model = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
        resp = genai.embed_content(model=model, content=text)

        vec: Optional[np.ndarray] = None

        # Compatibilidad 0.7.x (dict) y 0.8.x (objeto)
        if isinstance(resp, dict):
            emb_obj = resp.get("embedding")
            if isinstance(emb_obj, dict) and "values" in emb_obj:
                vec = np.array(emb_obj["values"], dtype=np.float32)
            elif isinstance(emb_obj, list):
                vec = np.array(emb_obj, dtype=np.float32)
        else:
            emb_obj = getattr(resp, "embedding", None)
            if isinstance(emb_obj, (list, tuple)):
                vec = np.array(emb_obj, dtype=np.float32)
            else:
                embeddings = getattr(resp, "embeddings", None)
                if embeddings and len(embeddings) > 0:
                    first = embeddings[0]
                    values = getattr(first, "values", None)
                    if values:
                        vec = np.array(values, dtype=np.float32)

        if vec is None:
            raise RuntimeError(f"Formato de respuesta de Gemini no reconocido: {type(resp)} -> {resp}")

        logger.info(f"[embed] provider=gemini model={model} -> dim_query={len(vec)} vs index_dim={self.dim}")
        return vec

    def _embed_openai(self, text: str) -> np.ndarray:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Falta OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)

        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        resp = client.embeddings.create(model=model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        logger.info(f"[embed] provider=openai model={model} -> dim_query={len(vec)} vs index_dim={self.dim}")
        return vec

    def embed_text(self, text: str) -> np.ndarray:
        # Forzamos proveedor según la dim del índice
        provider = "gemini" if self.dim == 768 else "openai"
        logger.info(f"[RAG] Using provider={provider} for query embeddings")

        if provider == "gemini":
            return self._embed_gemini(text)
        else:
            return self._embed_openai(text)

    # ---------- Búsqueda ----------
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
        logger.info("[RAG] Índice cargado desde R2.")
    except Exception as e:
        logger.warning(f"[RAG] No se pudo cargar desde R2: {e}. Probando local…")
        emb, meta = _load_local_index()
        logger.info("[RAG] Índice cargado desde disco local.")
    _INDEX = RAGIndex(embeddings=emb, metadata=meta)
    return _INDEX

# -------------
# Facades de búsqueda
# -------------
def retrieve_by_text(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    hits = get_index().search_by_text(query, top_k=top_k)
    logger.info(f"[RAG] query='{query[:80]}' -> hits={len(hits)}")
    return hits

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
