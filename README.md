# Proyecto Voto Informado CR — Starter (Flask/FastAPI + RAG)

## Requisitos
- Python 3.10+
- (Opcional) virtualenv

## Pasos
1. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. Copiá `.env.example` a `.env` y reemplazá `OPENAI_API_KEY` y otros valores.
4. Colocá tus PDFs en `app/data/pdf/` con formato `partido__titulo.pdf`.
5. `make ingest` para construir el índice (FAISS).
6. `make dev` para levantar la API en `http://localhost:8000`.
7. Abrí `http://localhost:8000/` (servidas las vistas Jinja desde FastAPI).

## Notas
- El chat responde **solo** con lo que esté en los planes (RAG) y **cita fuentes**.
- Anuncios (AdSense) están incluidos con rótulo **Publicidad**, reemplazá `ca-pub-XXXXXXXXXXXXXXX` + `data-ad-slot`.
- Cumplimiento TSE: respuestas neutrales, sin sensacionalismo; página `/acerca` con política editorial.

