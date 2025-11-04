# app/ingest_openai.py
import os, json, numpy as np
from pypdf import PdfReader
from openai import OpenAI

# === Rutas robustas ===
BASE_DIR = os.path.dirname(__file__)            # .../app
DOCS_DIR = os.path.join(BASE_DIR, "data", "pdfs")
OUT_DIR  = os.path.join(BASE_DIR, "data", "index")

MODEL    = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
BATCH    = 100
CHUNK    = 900
OVERLAP  = 150

def chunk_text(t, size=CHUNK, overlap=OVERLAP):
    t = (t or "").strip()
    if not t: return []
    step = max(1, size - overlap)
    return [t[i:i+size] for i in range(0, len(t), step)]

def parse_party_title(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    if "__" in base:
        party, title = base.split("__", 1)
    else:
        party, title = "desconocido", base
    return party.strip(), title.strip()

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno.")

    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR,  exist_ok=True)
    client = OpenAI(api_key=api_key)

    pdfs = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        raise RuntimeError(f"No hay PDFs en {DOCS_DIR}")

    meta, chunks = [], []
    for fn in pdfs:
        path = os.path.join(DOCS_DIR, fn)
        party, title = parse_party_title(fn)
        reader = PdfReader(path)
        for pi, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text: continue
            for ck in chunk_text(text):
                meta.append({"party": party, "title": title, "page": pi, "source": fn, "chunk": ck})
                chunks.append(ck)

    print(f"Total chunks: {len(chunks)}")

    embeds = []
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        resp = client.embeddings.create(model=MODEL, input=batch)
        embeds.extend([d.embedding for d in resp.data])

    arr = np.array(embeds, dtype=np.float32)
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), arr)
    with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print("Embeddings shape:", arr.shape)  # (N, 1536)

if __name__ == "__main__":
    main()
