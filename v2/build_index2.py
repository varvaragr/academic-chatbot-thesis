import os
import json
import sqlite3
import hashlib
from langchain_huggingface import HuggingFaceEmbeddings

DATA_FILE = os.getenv("DATA_FILE_V2", "base2.json")
DB_FILE = os.getenv("DB_FILE_V2", "chunks_v2_faculty.db")
EMBED_MODEL = os.getenv("EMBED_MODEL_V2", "Alibaba-NLP/gte-multilingual-base")
CHUNK_SIZE = 500
BATCH_SIZE = int(os.getenv("BATCH_SIZE_V2", "16"))

SKIP_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".mp4", ".avi", ".mov", ".zip", ".rar",
)

print("Initializing V2 embeddings...")
print(f"Model: {EMBED_MODEL}")
emb_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"trust_remote_code": True},
)
print("V2 embedding model loaded")

conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    text TEXT,
    source TEXT,
    embedding BLOB,
    hash TEXT
)
""")
conn.commit()

c.execute("PRAGMA table_info(chunks)")
cols = [col[1] for col in c.fetchall()]
if "hash" not in cols:
    c.execute("ALTER TABLE chunks ADD COLUMN hash TEXT")
    conn.commit()

print(f"Reading {DATA_FILE}...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)
print(f"Loaded {len(docs)} documents")

all_chunks = []
skipped_files = 0
for doc in docs:
    text = doc.get("text", "").strip()
    url = doc.get("url", "").strip()
    if not text:
        continue
    if url.lower().endswith(SKIP_EXTENSIONS):
        skipped_files += 1
        continue
    if len(text) < 50:
        continue

    # V2 structured chunking:
    # keep compact logical records intact; keep the original 500-char
    # chunking for ordinary pages and oversized records.
    is_structured_record = (
        "#v2-structured" in url
        or "#v2-faculty-courses" in url
        or "/semester/" in url
        or "cguide-25-26.pdf#page=" in url
    )
    structured_max_chars = 4000

    if is_structured_record and len(text) <= structured_max_chars:
        chunks_for_doc = [text]
    else:
        chunks_for_doc = [
            text[i:i + CHUNK_SIZE]
            for i in range(0, len(text), CHUNK_SIZE)
        ]

    for chunk_text in chunks_for_doc:
        chunk_hash = hashlib.md5(
            (chunk_text + url + EMBED_MODEL).encode("utf-8")
        ).hexdigest()
        all_chunks.append({"text": chunk_text, "source": url, "hash": chunk_hash})

print(f"Skipped files: {skipped_files}")
print(f"Split into {len(all_chunks)} chunks")

c.execute("SELECT hash FROM chunks")
existing_hashes = {row[0] for row in c.fetchall()}
new_chunks = [ch for ch in all_chunks if ch["hash"] not in existing_hashes]
print(f"New chunks to index: {len(new_chunks)}")

for start in range(0, len(new_chunks), BATCH_SIZE):
    batch = new_chunks[start:start + BATCH_SIZE]
    texts = [b["text"] for b in batch]
    embeddings = emb_model.embed_documents(texts)

    rows = [
        (batch[i]["text"], batch[i]["source"], json.dumps(emb), batch[i]["hash"])
        for i, emb in enumerate(embeddings)
    ]
    c.executemany(
        "INSERT INTO chunks (text, source, embedding, hash) VALUES (?, ?, ?, ?)",
        rows
    )
    conn.commit()
    print(f"Indexed {start + len(batch)} / {len(new_chunks)}")

conn.close()
print(f"Finished! V2 vector DB ready: {DB_FILE}")
