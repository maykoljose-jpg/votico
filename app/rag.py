import os, json
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import httpx

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "app/data"))
INDEX_DIR = DATA_DIR / "index"

EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "6"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Carga índice (NumPy)
emb_path = INDEX_DIR / "embeddings.npy"
meta_path = INDEX_DIR / "metadata.json"
if emb_path.exists() and meta_path.exists():
    _E = np.load(emb_path)            # (N, D) normalizado
    with open(meta_path, "r", encoding="utf-8") as f:
        _M = json.load(f)
else:
    _E = None
    _M = []

def search(query, k=TOP_K):
    if _E is None or not len(_M):
        return []
    q = _embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]  # (D,)
    # Cosine sim = dot product (por estar normalizados)
    scores = _E @ q  # (N,)
    # Top-K
    if k >= len(scores):
        idx = np.argsort(-scores)
    else:
        idx_part = np.argpartition(-scores, k)[:k]
        idx = idx_part[np.argsort(-scores[idx_part])]
    results = []
    for i in idx[:k]:
        m = dict(_M[int(i)])
        m["score"] = float(scores[int(i)])
        results.append(m)
    return results

SYSTEM_CONTENT = (
    "Sos un asistente neutral que responde EXCLUSIVAMENTE con base en los planes de gobierno provistos. "
    "Siempre citá partido, documento y página. Si no hay información suficiente, decí 'No se encontró en los planes'. "
    "No hagás afirmaciones que no estén respaldadas por los fragmentos."
)

def build_prompt(query, passages):
    context = ""
    for i, p in enumerate(passages, 1):
        context += (
            f"[{i}] Partido: {p['party']} | Doc: {p['title']} | Página: {p['page']}\n"
            f"{p['chunk']}\n\n"
        )
    user = (
        f"Pregunta del usuario: {query}\n\n"
        f"Contexto (fragmentos citables):\n{context}"
        "Instrucciones:\n"
        "- Responder de forma concisa y neutral.\n"
        "- Incluir sección 'Fuentes' con formato: Partido — Doc (p. X).\n"
        "- Si hay discrepancias, explicarlas sin tomar postura.\n"
    )
    return SYSTEM_CONTENT, user

async def generate_answer_openai(system, user):
    if not OPENAI_API_KEY:
        return "Error: falta OPENAI_API_KEY o no se creó el índice."
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

async def answer(query: str):
    passages = search(query)
    if not passages:
        return {"answer": "No hay índice construido o no se encontraron fragmentos para esta consulta.", "citations": []}
    system, user = build_prompt(query, passages)
    text = await generate_answer_openai(system, user)
    cites = [
        {"party": p["party"], "title": p["title"], "page": p["page"], "source": p["source"], "score": p["score"]}
        for p in passages
    ]
    return {"answer": text, "citations": cites}
import os, httpx

async def check_openai_connectivity():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"ok": False, "reason": "Falta OPENAI_API_KEY en .env"}
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=headers)
            # Mapeo simple de estados comunes
            if r.status_code == 200:
                return {"ok": True, "status": 200, "message": "Conexión OK y API key válida"}
            elif r.status_code == 401:
                return {"ok": False, "status": 401, "message": "API key inválida o malformada"}
            elif r.status_code == 429:
                return {"ok": False, "status": 429, "message": "Sin cuota / sin método de pago / límite alcanzado"}
            elif r.status_code == 403:
                return {"ok": False, "status": 403, "message": "Acceso bloqueado (verifica país/organización)"}
            else:
                return {"ok": False, "status": r.status_code, "message": r.text[:200]}
    except httpx.TimeoutException:
        return {"ok": False, "status": "timeout", "message": "Timeout conectando a api.openai.com"}
    except Exception as e:
        return {"ok": False, "status": "error", "message": str(e)}
