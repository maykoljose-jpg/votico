# app/ingest_semantic_gemini.py
# -*- coding: utf-8 -*-
"""
Ingesta semántica de PDFs + embeddings con Gemini (fallback a OpenAI).

- Extrae texto con PyMuPDF si está disponible; si no, usa pypdf.
- Segmenta por bloques semánticos (títulos/listas/párrafos), subdivide en frases
  y arma chunks de 900–1200 caracteres con solapamiento leve (120 chars).
- Genera embeddings con Google GenAI (text-embedding-004), o con OpenAI
  (text-embedding-3-small) si falta GOOGLE_API_KEY.
- Salida compatible con el RAG actual: embeddings.npy + metadata.json

ENV esperadas:
  PDF_DIR                (default: app/data/pdfs)
  OUT_DIR                (default: app/data/index)
  GOOGLE_API_KEY         (si está -> usa Gemini embeddings)
  GEMINI_EMBED_MODEL     (default: text-embedding-004)
  OPENAI_API_KEY         (fallback)
  OPENAI_EMBED_MODEL     (default: text-embedding-3-small)
"""

import os
import re
import json
import math
import glob
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm

# --------- Logging
logger = logging.getLogger("ingest")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ========= Extracción de texto =========

def _try_pymupdf_extract(pdf_path: str) -> List[str]:
    """Devuelve una lista de páginas (texto por página) usando PyMuPDF si existe."""
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []
    pages = []
    doc = fitz.open(pdf_path)
    for page in doc:
        txt = page.get_text("text")
        pages.append(txt)
    doc.close()
    return pages

def _pypdf_extract(pdf_path: str) -> List[str]:
    """Fallback con pypdf. Retorna una lista de textos por página."""
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("pypdf no está instalado y PyMuPDF falló: instala uno de los dos.") from e
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(txt)
    return pages

def extract_pages(pdf_path: str) -> List[str]:
    pages = _try_pymupdf_extract(pdf_path)
    if pages:
        return pages
    return _pypdf_extract(pdf_path)

# ========= Utilitarios de limpieza y segmentación =========

_re_heading = re.compile(r"^\s*(?:[0-9]+\.){1,3}\s+|^\s*(?:CAP[IÍ]TULO|T[ÍI]TULO|SECCI[ÓO]N)\b|^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s\-]{3,}$")
_re_bullet  = re.compile(r"^\s*[-•·▪◦]\s+|^\s*[0-9]+\)\s+")

def normalize_whitespace(text: str) -> str:
    # quita espacios raros, múltiples saltos, etc.
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def split_paragraphs(page_text: str) -> List[str]:
    """Separa por dobles saltos; mantiene listas/encabezados pegados a su párrafo siguiente si es corto."""
    blocks = [b.strip() for b in page_text.split("\n\n") if b.strip()]
    merged: List[str] = []
    i = 0
    while i < len(blocks):
        cur = blocks[i]
        if i+1 < len(blocks):
            nxt = blocks[i+1]
        else:
            nxt = ""
        # Si es un encabezado o bullet muy corto, júntalo con el siguiente
        if (_re_heading.search(cur) or _re_bullet.search(cur)) and len(cur) < 120 and nxt:
            merged.append((cur + " " + nxt).strip())
            i += 2
        else:
            merged.append(cur)
            i += 1
    return merged

def split_sentences(text: str) -> List[str]:
    """Segmentador simple por puntuación. Evita cortar en abreviaturas comunes."""
    # Protege abreviaturas simples
    text = re.sub(r"\b(p\.)\s?(ej\.)", r"p.ej.", text, flags=re.IGNORECASE)
    # Corta en . ! ? seguidos de espacio/mayúscula
    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-ZÁÉÍÓÚÑ(])", text)
    # Rejunta si quedaron micro oraciones
    out: List[str] = []
    buf = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        buf.append(s)
        if len(" ".join(buf)) >= 180:  # oraciones pequeñas se fusionan hasta aprox. ~180 chars
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out

def to_semantic_chunks(paragraphs: List[str],
                       target_min: int = 900,
                       target_max: int = 1200,
                       overlap_chars: int = 120) -> List[str]:
    """
    Junta frases de párrafos hasta target_min/target_max.
    Si un párrafo es muy largo, se divide por frases.
    Usa solapamiento en los bordes para mantener contexto.
    """
    chunks: List[str] = []
    cur: List[str] = []

    def flush():
        if cur:
            chunks.append(" ".join(cur).strip())

    for para in paragraphs:
        para = normalize_whitespace(para)
        if not para:
            continue
        if len(para) <= target_max:
            # Encaja tal cual (o lo acumulamos si no alcanzamos mínimo)
            if sum(len(x) for x in cur) + len(para) + len(cur) <= target_max:
                cur.append(para)
            else:
                # flush actual, iniciar nuevo
                if sum(len(x) for x in cur) >= max(280, target_min//2):  # evita chunks ridículos
                    flush()
                    # solapamiento: tomar últimos overlap_chars del chunk previo
                    if overlap_chars and chunks:
                        tail = chunks[-1][-overlap_chars:]
                        cur[:] = [tail + " " + para]
                    else:
                        cur[:] = [para]
                else:
                    # si lo acumulado es pequeño, forzamos corte por seguridad
                    flush()
                    cur[:] = [para]
        else:
            # párrafo demasiado grande -> dividir por frases
            sents = split_sentences(para)
            buf: List[str] = []
            for sent in sents:
                if sum(len(x) for x in buf) + len(sent) + len(buf) <= target_max:
                    buf.append(sent)
                else:
                    if buf:
                        chunk = " ".join(buf).strip()
                        if chunk:
                            if overlap_chars and chunks:
                                tail = chunks[-1][-overlap_chars:]
                                chunk = (tail + " " + chunk).strip()
                            chunks.append(chunk)
                    buf = [sent]
            if buf:
                chunk = " ".join(buf).strip()
                if chunk:
                    if overlap_chars and chunks:
                        tail = chunks[-1][-overlap_chars:]
                        chunk = (tail + " " + chunk).strip()
                    chunks.append(chunk)

            # reinicia acumulador superior
            cur = []

        # si ya excedimos target_min bastante, hacemos flush
        tot = sum(len(x) for x in cur) + len(cur) - 1
        if tot >= target_min:
            flush()
            cur = []

    flush()
    # limpieza final: descarta restos muy cortos y duplicados
    final = []
    seen = set()
    for ch in chunks:
        ch = ch.strip()
        if len(ch) < 240:
            continue
        key = ch[:80]
        if key in seen:
            continue
        seen.add(key)
        final.append(ch)
    return final

# ========= Embeddings (Gemini / OpenAI) =========

def embed_batch_texts(texts: List[str]) -> np.ndarray:
    """
    Calcula embeddings para una lista de textos.
    - Si hay GOOGLE_API_KEY -> Gemini text-embedding-004
    - Si no -> OpenAI text-embedding-3-small
    Devuelve ndarray (N, D)
    """
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if google_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_key)
            model = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
            vecs = []
            for t in tqdm(texts, desc="Gemini embeddings"):
                # La API de embeddings de Gemini acepta 1 texto por llamada (a la fecha).
                emb = genai.embed_content(model=model, content=t, task_type="RETRIEVAL_DOCUMENT")
                v = emb["embedding"]
                vecs.append(np.asarray(v, dtype=np.float32))
            arr = np.vstack(vecs)
            return arr
        except Exception as e:
            logger.warning(f"Fallo Gemini, uso OpenAI: {e}")

    # Fallback: OpenAI
    import openai  # paquete 'openai' >= 1.x (si tenés 0.x, ajusta)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("No GOOGLE_API_KEY ni OPENAI_API_KEY: no puedo calcular embeddings.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    vecs = []
    # OpenAI permite batch, pero para simplicidad lo hacemos 1x1 (estable y claro)
    for t in tqdm(texts, desc="OpenAI embeddings"):
        resp = client.embeddings.create(model=model, input=t)
        vecs.append(np.asarray(resp.data[0].embedding, dtype=np.float32))
    arr = np.vstack(vecs)
    return arr

# ========= Proceso principal =========

def process_pdf(pdf_path: str,
                party: str,
                title: str,
                target_min: int = 900,
                target_max: int = 1200,
                overlap_chars: int = 120) -> Tuple[List[str], List[Dict[str, Any]]]:
    pages = extract_pages(pdf_path)
    paragraphs: List[str] = []
    for p_i, page_txt in enumerate(pages, start=1):
        page_txt = normalize_whitespace(page_txt)
        if not page_txt:
            continue
        paragraphs.extend(split_paragraphs(page_txt))

    chunks = to_semantic_chunks(paragraphs, target_min, target_max, overlap_chars)

    meta: List[Dict[str, Any]] = []
    for idx, ch in enumerate(chunks, start=1):
        meta.append({
            "party": party or "desconocido",
            "title": title,
            "chunk": ch,
            "page": None,   # opcional: podríamos mapear a páginas si guardamos anclas
            "source": os.path.basename(pdf_path),
        })
    return chunks, meta

def guess_party_and_title(filename: str) -> Tuple[str, str]:
    name = os.path.splitext(os.path.basename(filename))[0]
    # ejemplo de heurística simple:
    # "Frente Amplio_ Plan de Gobierno.pdf" -> party="Frente Amplio", title="Plan de Gobierno"
    name = name.replace("_", " ").replace("-", " ").strip()
    if "plan" in name.lower():
        parts = name.split("plan")
        party = parts[0].strip(" _-—")
        title = "Plan " + "plan".join(parts[1:]).strip(" _-—")
        if not title:
            title = "Plan de Gobierno"
        return party, title
    # default:
    return name, "Plan de Gobierno"

def main():
    PDF_DIR = os.getenv("PDF_DIR", "app/data/pdfs")
    OUT_DIR = os.getenv("OUT_DIR", "app/data/index")
    os.makedirs(OUT_DIR, exist_ok=True)

    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdfs:
        logger.error(f"No hay PDFs en {PDF_DIR}")
        return

    all_texts: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    for pdf_path in pdfs:
        party, title = guess_party_and_title(pdf_path)
        logger.info(f"Procesando: {os.path.basename(pdf_path)} -> {party} / {title}")
        chunks, meta = process_pdf(pdf_path, party=party, title=title,
                                   target_min=900, target_max=1200, overlap_chars=120)
        all_texts.extend(chunks)
        all_meta.extend(meta)

    logger.info(f"Total chunks: {len(all_texts)}")
    # Embeddings
    emb = embed_batch_texts(all_texts)   # (N, D)

    # Guardar
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), emb)
    with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    logger.info(f"Listo -> {OUT_DIR}/embeddings.npy + metadata.json (shape={emb.shape})")

if __name__ == "__main__":
    main()
