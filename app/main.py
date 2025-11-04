# app/main.py
import os, traceback, uuid
from collections import deque, defaultdict
from typing import Deque, Dict, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader, select_autoescape
from datetime import datetime
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from .rag import answer, check_openai_connectivity, index_stats  # noqa

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

# ========= Memoria por sesión =========
MAX_TURNS = int(os.getenv("CHAT_MEMORY_MESSAGES", "5"))
CHAT_MEMORY: Dict[str, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=2*MAX_TURNS))
# =====================================

class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    use_memory: bool = True

# ------------------- Páginas -------------------

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

# ------------------- APIs util -------------------

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

# ------------------- Chat RAG -------------------

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        q = (req.query or "").strip()
        if not q:
            return {"answer": "Escribí una pregunta.", "citations": []}

        sid = req.session_id or str(uuid.uuid4())
        history = list(CHAT_MEMORY[sid]) if req.use_memory else None

        resp = await answer(q, history=history)

        if not isinstance(resp, dict):
            resp = {"answer": str(resp), "citations": []}

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
            except:
                continue
        resp["citations"] = cites
        resp["answer"] = str(resp.get("answer",""))

        if req.use_memory and resp.get("answer"):
            CHAT_MEMORY[sid].append({"role": "user", "content": q})
            CHAT_MEMORY[sid].append({"role": "assistant", "content": resp["answer"]})

        resp["session_id"] = sid
        resp["turns_kept"] = len(CHAT_MEMORY[sid])
        return resp
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Chat backend error: {e}")
