# app/main.py
# -*- coding: utf-8 -*-
import os
from datetime import datetime
from typing import List, Optional, Literal

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv, find_dotenv
import httpx

# Cargar .env local (Render también inyecta como vars de entorno)
load_dotenv(find_dotenv())

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

# CORS
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Navegación/Pages ----
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
    return render_template("partidos.html", year=datetime.now().year, partidos=PARTIDOS, q=q or "", request=request)

@app.get("/contacto", response_class=HTMLResponse)
async def contacto(request: Request):
    return render_template("contacto.html", year=datetime.now().year, request=request)

@app.get("/acerca", response_class=HTMLResponse)
async def acerca(request: Request):
    return render_template("acerca.html", year=datetime.now().year, request=request)

# ---- Modelos de API ----
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: Optional[str] = None
    history: Optional[List[ChatTurn]] = None  # historial opcional

# Importamos después de crear 'app' para evitar import circular
from .rag import answer, check_openai_connectivity, index_stats

# ---- Endpoints utilitarios ----
@app.get("/api/health")
async def health():
    return {"ok": True}

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

@app.get("/api/openai-check")
async def openai_check():
    return await check_openai_connectivity()

@app.get("/api/index-stats")
async def api_index_stats():
    return index_stats()

# Echo simple a OpenAI (debug)
@app.get("/api/openai-echo")
async def openai_echo():
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return {"status": 400, "where": "env", "error": "Falta OPENAI_API_KEY"}
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

# ---- Chat principal (con historial) ----
@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        q = (req.query or "").strip()
        if not q:
            return {"answer": "Escribí una pregunta.", "citations": [], "session_id": req.session_id}

        # Sanitizar/recortar historial (máx 10 turnos recientes)
        history = []
        if req.history:
            for t in req.history[-10:]:
                history.append({"role": t.role, "content": t.content})

        resp = await answer(q, history=history)
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

        return {
            "answer": str(resp.get("answer","")),
            "citations": cites,
            "session_id": req.session_id,
        }

    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chat backend error: {e}")
