import os, json, time, gc
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()
DATA_DIR   = Path(os.getenv("DATA_DIR", "app/data"))
PDF_DIR    = DATA_DIR / "pdf"
INDEX_DIR  = DATA_DIR / "index"
WORK_DIR   = INDEX_DIR / "_work"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Parámetros conservadores para bajar uso de RAM
MAX_CHARS_PER_CHUNK   = int(os.getenv("MAX_CHARS_PER_CHUNK", "1400"))
CHUNK_OVERLAP         = int(os.getenv("CHUNK_OVERLAP", "100"))
BATCH_SIZE            = int(os.getenv("EMBED_BATCH", "32"))  # si falta RAM, probá 16
MAX_TEXT_CHARS_PER_PG = int(os.getenv("MAX_TEXT_CHARS_PER_PG", "200000"))  # fusible para PDFs raros


def chunk_gen(text: str, max_chars=MAX_CHARS_PER_CHUNK, overlap=CHUNK_OVERLAP):
    """Generador de chunks sin crear listas gigantes."""
    n = len(text)
    i = 0
    while i < n:
        end = min(i + max_chars, n)
        yield text[i:end]
        if end == n:
            break
        i = end - overlap if end - overlap > i else end


def write_chunks_jsonl():
    """Lee PDFs y escribe cada chunk como una línea JSON en WORK_DIR/chunks.jsonl"""
    out_path = WORK_DIR / "chunks.jsonl"
    if out_path.exists():
        out_path.unlink()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"🔎 PDFs encontrados: {len(pdfs)}")
    count = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for pdf in pdfs:
            t0 = time.time()
            name = pdf.stem
            if "__" in name:
                party, title = name.split("__", 1)
            else:
                party, title = "desconocido", name

            reader = PdfReader(str(pdf))
            total_pages = len(reader.pages)
            print(f"📄 {pdf.name} — {total_pages} páginas")

            for pnum, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                # Fusible para páginas con extracción anómala
                if len(text) > MAX_TEXT_CHARS_PER_PG:
                    text = text[:MAX_TEXT_CHARS_PER_PG]

                for ch in chunk_gen(text):
                    obj = {
                        "party": party,
                        "title": title,
                        "page":  pnum,
                        "chunk": ch,
                        "source": pdf.name
                    }
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    count += 1

                if pnum % 10 == 0:
                    print(f"  · Página {pnum}/{total_pages} (chunks: {count})")

            print(f"✅ {pdf.name} listo en {time.time()-t0:.1f}s")

    print(f"🧾 chunks.jsonl creado con {count} líneas")
    return out_path, count


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def build_embeddings(jsonl_path: Path):
    """Segunda pasada: embebe en lotes y escribe a un memmap sin cargar todo en RAM."""
    model = SentenceTransformer(EMBED_MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"🧠 Modelo: {EMBED_MODEL_NAME} (dim={dim})")

    total = count_lines(jsonl_path)
    print(f"📦 Total de chunks: {total}")

    # Memmap para no consumir RAM
    mmap_path = INDEX_DIR / "embeddings.dat"
    E = np.memmap(mmap_path, dtype="float32", mode="w+", shape=(total, dim))

    meta = []  # lo escribiremos al final como JSON
    t0 = time.time()

    with jsonl_path.open("r", encoding="utf-8") as f:
        i = 0
        batch_texts = []
        batch_meta = []
        for line in f:
            d = json.loads(line)
            txt = d.get("chunk") or ""
            batch_texts.append(txt)
            batch_meta.append({
                "party":  d["party"],
                "title":  d["title"],
                "page":   d["page"],
                "source": d["source"],
                "chunk":  txt
            })

            if len(batch_texts) >= BATCH_SIZE:
                embs = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
                E[i:i+len(embs)] = embs
                meta.extend(batch_meta)
                i += len(embs)
                batch_texts.clear(); batch_meta.clear()
                if i % (BATCH_SIZE*10) == 0:
                    print(f"  · Embeddings {i}/{total}")
                gc.collect()

        # Último batch
        if batch_texts:
            embs = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
            E[i:i+len(embs)] = embs
            meta.extend(batch_meta)
            i += len(embs)

    # Aseguramos escritura a disco
    E.flush(); del E

    # Convertimos .dat (memmap) a .npy final
    arr = np.memmap(mmap_path, dtype="float32", mode="r", shape=(total, dim))
    np.save(INDEX_DIR / "embeddings.npy", np.array(arr))
    del arr
    try:
        mmap_path.unlink()
    except Exception:
        pass

    # 🔴 IMPORTANTE: metadata.json con "chunk" incluido
    with (INDEX_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump([
            {
                "party": m["party"],
                "title": m["title"],
                "page":  m["page"],
                "source": m["source"],
                "chunk": m.get("chunk","")
            }
            for m in meta
        ], f, ensure_ascii=False)

    print(f"🎉 Índice listo en {time.time()-t0:.1f}s → embeddings.npy + metadata.json")


if __name__ == "__main__":
    if not any(PDF_DIR.glob("*.pdf")):
        print("⚠️ No se encontraron PDFs en app/data/pdf/")
        raise SystemExit(0)
    jsonl_path, _ = write_chunks_jsonl()
    build_embeddings(jsonl_path)
