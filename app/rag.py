# app/rag.py
# -*- coding: utf-8 -*-
"""
RAG minimalista + respuesta conversacional:
- Carga embeddings y metadata desde Cloudflare R2 (público) o del disco local.
- Mantiene el índice en caché en memoria (normalizado) para similitud coseno.
- Ofrece búsqueda por texto (embedding vía OpenAI) o directamente por vector.
- Implementa `answer()` con tono conversacional, citas [n] y fallbacks robustos.

Requiere:
  pip install numpy httpx openai

Variables de entorno clave:
  R2_PUBLIC_BASE           -> p.ej. https://pub-xxxxxxxxxxxxxxxxxxxxx.r2.dev
  INDEX_PREFIX             -> p.ej. index (default: index)
  OPENAI_API_KEY           -> tu API key
  OPENAI_EMBED_MODEL       -> default: text-embedding-3-small
  OPENAI_MODEL             -> default: gpt-4o-mini
  TOP_K                    -> default: 6
  ANSWER_TEMPERATURE       -> default: 0.35
  ANSWER_MAX_TOKENS        -> default: 600
  MOCK_MODE                -> "1" para modo demo (no llama LLM)
"""
# --- Embeddings de consulta con SentenceTransformers (mismo que el índice)
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False

_QUERY_MODEL = None
def _get_query_model():
    global _QUERY_MODEL
    if _QUERY_MODEL is None:
        name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        if not _ST_AVAILABLE:
            raise RuntimeError("Falta sentence-transformers para embebidos de consulta.")
        _QUERY_MODEL = SentenceTransformer(name)
    return _QUERY_MODEL

def embed_query_locally(text: str) -> np.ndarray:
    model = _get_query_model()
    v = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    return v

from __future__ import annotations

import json
import os
import logging
from io import BytesIO
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import httpx

# OpenAI opcional (solo si vas a usar search_by_text)
try:
    from openai import OpenAI  # SDK nuevo
    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    _OPENAI_AVAILABLE = False

# ---- Logging sencillo
logger = logging.getLogger("rag")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# =========================
#   Utilidades de descarga
# =========================

def _join_public_url(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    tail = "/".join(p.strip("/") for p in parts if p and p != "/")
    return f"{base}/{tail}"


def _fetch_bytes(url: str, timeout: int = 30) -> bytes:
    """Descarga un recurso binario con httpx (streaming protegido)."""
    logger.info(f"Descargando: {url}")
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.content


def _load_remote_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Intenta descargar embeddings.npy y metadata.json desde R2 público."""
    base = os.getenv("R2_PUBLIC_BASE", "").strip()
    if not base:
        raise RuntimeError("R2_PUBLIC_BASE no está definido.")

    prefix = os.getenv("INDEX_PREFIX", "index").strip().strip("/")
    meta_url = _join_public_url(base, prefix, "metadata.json")
    emb_url = _join_public_url(base, prefix, "embeddings.npy")

    metadata_bytes = _fetch_bytes(meta_url)
    embeddings_bytes = _fetch_bytes(emb_url)

    metadata = json.loads(metadata_bytes.decode("utf-8"))
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
        raise FileNotFoundError(
            f"No se encontró el índice local en {base_dir}"
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    embeddings = np.load(emb_path, allow_pickle=False)
    return embeddings, metadata


# =========================
#       RAGIndex
# =========================

class RAGIndex:
    """Índice RAG en memoria con embeddings normalizados para búsqueda rápida."""

    def __init__(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        if embeddings.ndim != 2:
            raise ValueError("Embeddings deben ser un array 2D (N, D).")
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"metadata ({len(metadata)}) no coincide con embeddings ({embeddings.shape[0]})."
            )

        # Convertimos a float32 para ahorrar memoria
        self._emb = embeddings.astype(np.float32, copy=False)

        # Normalizamos fila a fila para similitud coseno rápida
        norms = np.linalg.norm(self._emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._emb = self._emb / norms

        self._meta = metadata
        self.dim = self._emb.shape[1]
        logger.info(f"RAGIndex cargado: {self._emb.shape[0]} vectores, dim={self.dim}")

    @property
    def size(self) -> int:
        return self._emb.shape[0]

    def search_by_vector(
        self, query_vec: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Busca por vector (ya embebido). Retorna lista de dicts:
          {score, metadata}
        """
        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim != 1:
            raise ValueError("query_vec debe ser 1D (dim,).")
        if q.shape[0] != self.dim:
            raise ValueError(f"Dim mismatch: query={q.shape[0]} vs index={self.dim}")

        # Normalizamos el query para similitud coseno
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        # Similaridad coseno como producto punto de vectores normalizados
        sims = self._emb @ q_norm  # (N,)

        # Top-K más altos
        top_k = max(1, min(int(top_k), self.size))
        idx = np.argpartition(-sims, top_k - 1)[:top_k]
        # Ordenar los top_k por score desc
        idx = idx[np.argsort(-sims[idx])]

        results = []
        for i in idx:
            results.append({
                "score": float(sims[i]),
                "metadata": self._meta[i],
            })
        return results

    # -------- OpenAI helpers (opcional) --------
    def _ensure_openai(self) -> OpenAI:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError(
                "Paquete 'openai' no disponible. Instala 'openai' o usa search_by_vector()."
            )
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no está definido.")
        return OpenAI(api_key=api_key)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed una consulta de texto con OpenAI (modelo small por defecto)."""
        client = self._ensure_openai()
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        resp = client.embeddings.create(model=model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return vec

    def search_by_text(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Embed + búsqueda."""
        vec = self.embed_text(text)
        return self.search_by_vector(vec, top_k=top_k)


# =========================
#  Carga y caché del índice
# =========================

_INDEX: Optional[RAGIndex] = None  # caché en módulo


def get_index(force_reload: bool = False) -> RAGIndex:
    """
    Devuelve el índice cargado (y cacheado).
    - Intenta primero R2 público (si R2_PUBLIC_BASE está presente).
    - Si falla, hace fallback a disco local.
    """
    global _INDEX
    if _INDEX is not None and not force_reload:
        return _INDEX

    try:
        embeddings, metadata = _load_remote_index()
        logger.info("Índice cargado desde Cloudflare R2.")
    except Exception as e_remote:
        logger.warning(f"No se pudo cargar desde R2: {e_remote}. Probando local...")
        embeddings, metadata = _load_local_index()
        logger.info("Índice cargado desde disco local.")

    _INDEX = RAGIndex(embeddings=embeddings, metadata=metadata)
    return _INDEX


# =========================
#  Facades de alto nivel
# =========================


def retrieve_by_text(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Búsqueda rápida por texto usando el MISMO modelo que generó el índice.
    Retorna: [{score, metadata}, ...]
    """
    index = get_index()
    # Embebido local (384 dims por defecto con all-MiniLM-L6-v2)
    vec = embed_query_locally(query)
    return index.search_by_vector(vec, top_k=top_k)


def retrieve_by_vector(query_vec: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Búsqueda por vector (si ya embebiste en otra capa).
    Retorna: [{score, metadata}, ...]
    """
    index = get_index()
    return index.search_by_vector(np.asarray(query_vec, dtype=np.float32), top_k=top_k)


def top_k_context(query: str, top_k: int = 5, key_field: str = "text") -> List[str]:
    """
    Atajo: devuelve SOLO los textos (o el campo elegido) del top-k.
    Útil para armar el prompt al modelo de chat.
    """
    hits = retrieve_by_text(query, top_k=top_k)
    ctx = []
    for h in hits:
        md = h["metadata"] or {}
        # Intentamos distintos nombres comunes
        text = md.get(key_field) or md.get("chunk") or md.get("content") or ""
        ctx.append(text)
    return ctx


# =========================
#  Conversational answer()
# =========================

def _format_sources(hits: List[Dict[str, Any]]) -> str:
    """Convierte los top-k en lista numerada de fuentes + texto corto."""
    lines = []
    for i, h in enumerate(hits, start=1):
        md = h.get("metadata", {}) or {}
        party = md.get("party", "")
        title = md.get("title", "")
        page = md.get("page", "")
        src  = md.get("source", "")
        chunk = (md.get("chunk") or md.get("text") or md.get("content") or "").strip()
        snippet = chunk[:600].replace("\n", " ")
        lines.append(f"[{i}] ({party}) {title} – pág. {page} – {src}\n>>> {snippet}")
    return "\n\n".join(lines)


def _build_messages(query: str, hits: List[Dict[str, Any]], style: str = "CONVERSATIONAL") -> List[Dict[str, str]]:
    """
    Arma un prompt que: 1) responde con criterio, 2) cita fuentes [1][2],
    3) hace una pregunta de seguimiento.
    """
    context_block = _format_sources(hits)
    guidance = (
        "Sos un asistente neutral para votar en Costa Rica.\n"
        "RESPONDE usando SOLO la evidencia del contexto. Si algo no está en las fuentes, decí que no aparece.\n"
        "Estilo: claro, breve, conversacional; explica el porqué en lenguaje simple, sin listas largas.\n"
        "Incluí citas tipo [1], [2] en las frases que dependen de cada fuente.\n"
        "Si hay diferencias entre partidos, compáralas en 2–4 oraciones.\n"
        "Cerrá con UNA pregunta de seguimiento."
    )
    if style.upper() == "TUTOR":
        guidance += "\nUsá preguntas socráticas breves para guiar al usuario."

    user_prompt = (
        f"Pregunta del usuario:\n{query}\n\n"
        f"Contexto (fragmentos de planes de gobierno):\n{context_block}\n\n"
        "Tarea: redactá una respuesta conversacional y precisa, con citas [n]."
    )
    return [
        {"role": "system", "content": guidance},
        {"role": "user", "content": user_prompt},
    ]


async def _openai_chat(messages, model: str, temperature: float, max_tokens: int, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()


async def answer(query: str, top_k: int = None) -> Dict[str, Any]:
    """
    Conversational RAG:
    - Recupera top-k fragmentos
    - Construye un prompt con guía conversacional y citas
    - Llama a OpenAI y retorna {answer, citations}
    - Fallbacks robustos si no hay API key / error de red
    """
    # 1) Recuperación
    k = int(top_k or os.getenv("TOP_K", "6"))
    hits = retrieve_by_text(query, top_k=k)

    # 2) Si NO hay ni un hit, respondé claro
    if not hits:
        return {"answer": "Busqué en los planes y no vi propuestas relevantes sobre ese tema.", "citations": []}

    # 3) Siempre armamos las citas (aunque el LLM falle)
    citations = []
    for h in hits:
        md = h.get("metadata", {}) or {}
        citations.append({
            "party":  str(md.get("party","")),
            "title":  str(md.get("title","")),
            "page":   md.get("page",""),
            "source": str(md.get("source","")),
            "score":  float(h.get("score", 0.0)),
        })

    # 4) Modo demo
    if os.getenv("MOCK_MODE", "0") == "1":
        return {
            "answer": "Resumen breve y conversacional basado en los fragmentos recuperados [1][2]. "
                      "¿Querés que compare por partido?",
            "citations": citations
        }

    # 5) Config OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        # Sin API key, devolvé un resumen mínimo con las fuentes (mejor que 'no hay')
        bullets = []
        for i, h in enumerate(hits, 1):
            md = h.get("metadata", {}) or {}
            t = (md.get("chunk") or md.get("text") or "")[:200].replace("\n"," ")
            bullets.append(f"[{i}] {t}")
        return {
            "answer": "Encontré fragmentos relevantes en los planes, pero no pude generar el texto con el modelo. "
                      "Acá tenés un resumen mínimo:\n- " + "\n- ".join(bullets) + "\n\n¿Querés que intente de nuevo?",
            "citations": citations
        }

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("ANSWER_TEMPERATURE", "0.35"))
    max_tokens = int(os.getenv("ANSWER_MAX_TOKENS", "600"))
    messages = _build_messages(query, hits, style=os.getenv("ANSWER_STYLE","CONVERSATIONAL"))

    # 6) LLM con fallback
    try:
        text = await _openai_chat(messages, model, temperature, max_tokens, api_key)
    except Exception as e:
        logger.warning(f"Fallo al llamar OpenAI: {e}")
        bullets = []
        for i, h in enumerate(hits, 1):
            md = h.get("metadata", {}) or {}
            t = (md.get("chunk") or md.get("text") or "")[:200].replace("\n"," ")
            bullets.append(f"[{i}] {t}")
        return {
            "answer": "No pude conectarme al modelo en este momento, pero sí hay material en los planes. "
                      "Resumen rápido de los fragmentos:\n- " + "\n- ".join(bullets) +
                      "\n\n¿Te comparo por partido o querés que lo intente otra vez?",
            "citations": citations
        }

    return {"answer": text, "citations": citations}


# =========================
#  Diagnóstico / utilidades
# =========================

async def check_openai_connectivity() -> Dict[str, Any]:
    """Usado por /api/openai-check en main.py para validar la clave y el modelo."""
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
    """Estadísticas rápidas del índice cargado en memoria."""
    idx = get_index()
    from collections import Counter
    parties = Counter([(m.get("party") or "desconocido") for m in getattr(idx, "_meta", [])])
    return {"vectors": idx.size, "dim": idx.dim, "parties": dict(parties)}
