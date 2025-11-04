# app/main.py
import os
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

# Cargar variables de entorno (.env) al iniciar
load_dotenv(find_dotenv())

# Importar SOLO una vez, y siempre al tope (evita imports circulares)
from .rag import answer, check_openai_connectivity, index_stats

import httpx
import traceback

app = FastAPI(title="Voto Informado CR")

# Static
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates_env = Environment(
    loader=FileSystemLoader("app/templates"),
    autoescape=select_autoescape(["html", "xml"]),
)

def render_template(name: str, **context) -> HTMLResponse:
    template = templates_env.get_template(name)
    html = template.render(**context)
    return HTMLResponse(content=html)

# CORS (si el frontend está en otro dominio)
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PARTIDOS = [
    {"name": "Liberación Nacional (PLN)", "slug": "pln", "flag": "/static/assets/banderas/pln.svg"},
    {"name": "Unidad Social Cristiana (PUSC)", "slug": "pusc", "flag": "/static/assets/banderas/pusc.svg"},
    {"name": "Progreso Social Democrático (PPSD)", "slug": "ppsd", "flag": "/static/assets/banderas/ppsd.svg"},
    {"name": "Frente Amplio", "slug": "frente-amplio", "flag": "/static/assets/banderas/frente-amplio.svg"},
    {"name": "Nueva República", "slug": "nueva-republica", "flag": "/static/assets/banderas/nueva-republica.svg"},
    {"name": "Acción Ciudadana (PAC)", "slug": "pac", "flag": "/static/assets/banderas/pac.svg"},
]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    import random
    partidos = PARTIDOS[:]
    random.shuffle(partidos)
    return render_template("index.html", year=datetime.now().year, partidos=partidos, request=request)

@app.get("/partidos", response_class=HTMLResponse)
async def partidos(request: Request, q: str | None = None):
    items = PARTIDOS
    return render_template("partidos.html", year=datetime.now().year, partidos=items, q=q or "", request=request)

@app.get("/contacto", response_class=HTMLResponse)
async def contacto(request: Request):
    return render_template("contacto.html", year=datetime.now().year, request=request)

@app.get("/acerca", response_class=HTMLResponse)
async def acerca(request: Request):
    return render_template("acerca.html", year=datetime.now().year, request=request)

class ChatRequest(BaseModel):
    query: str

@app.get("/api/health")
async def health():
    return {"ok": True}

@app.get("/api/openai-echo")
async def openai_echo():
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return {"status": 400, "where": "env", "error": "Falta OPENAI_API_KEY en .env"}

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "Sos un asistente breve."},
            {"role": "user", "content": "Decí 'ping' y nada más."}
        ]
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, headers=headers, json=payload)
            return {"status": r.status_code, "body": r.text[:400]}
    except httpx.TimeoutException:
        return {"status": "timeout", "error": "Timeout conectando a api.openai.com"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/api/openai-check")
async def openai_check():
    return await check_openai_connectivity()

@app.get("/api/index-stats")
async def api_index_stats():
    return index_stats()

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        q = (req.query or "").strip()
        if not q:
            return {"answer": "Escribí una pregunta.", "citations": []}
        resp = await answer(q)
        if not isinstance(resp, dict):
            resp = {"answer": str(resp), "citations": []}

        # Normalizar citas
        cites = []
        for c in (resp.get("citations") or []):
            try:
                cites.append({
                    "party": str(c.get("party","")),
                    "title": str(c.get("title","")),
                    "page": c.get("page",""),
                    "source": str(c.get("source","")),
                    "score": float(c.get("score", 0.0)),
                })
            except Exception:
                continue
        resp["citations"] = cites
        resp["answer"] = str(resp.get("answer",""))
        return resp
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Chat backend error: {e}")

@app.get("/api/debug-env")
async def debug_env():
    return {
        "OPENAI_API_KEY_exists": bool(os.getenv("OPENAI_API_KEY")),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL"),
        "MOCK_MODE": os.getenv("MOCK_MODE"),
        "DATA_DIR": os.getenv("DATA_DIR"),
        "R2_PUBLIC_BASE": os.getenv("R2_PUBLIC_BASE"),
        "INDEX_PREFIX": os.getenv("INDEX_PREFIX"),
    }

