import os
import json
import sqlite3
import hashlib
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from typing import Dict, List


DATA_FILE = os.getenv("DATA_FILE", "ds_content.json")
DB_FILE = os.getenv("DB_FILE", "chunks.db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 500
BATCH_SIZE = 100

# Αρχεία που ΔΕΝ θέλουμε να μπουν στο RAG
SKIP_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".mp4",
    ".avi",
    ".mov",
    ".zip",
    ".rar",
)

print("Initializing embeddings...")
emb_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL
)
print("Embedding model loaded")


# Setup SQLite
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

# Ensure hash column exists
c.execute("PRAGMA table_info(chunks)")
cols = [col[1] for col in c.fetchall()]

if "hash" not in cols:
    c.execute("ALTER TABLE chunks ADD COLUMN hash TEXT")
    conn.commit()


# Load JSON
print(f"Reading {DATA_FILE}...")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)

print(f"Loaded {len(docs)} documents")


# Prepare chunks
all_chunks = []
skipped_files = 0

for doc in docs:

    text = doc.get("text", "").strip()
    url = doc.get("url", "").lower()

    if not text:
        continue

    # Skip εικόνες/αρχεία
    if url.endswith(SKIP_EXTENSIONS):
        skipped_files += 1
        continue

    # Skip πολύ μικρά chunks
    if len(text) < 50:
        continue

    for i in range(0, len(text), CHUNK_SIZE):

        chunk_text = text[i:i + CHUNK_SIZE]

        chunk_hash = hashlib.md5(
            (chunk_text + url).encode("utf-8")
        ).hexdigest()

        all_chunks.append({
            "text": chunk_text,
            "source": url,
            "hash": chunk_hash
        })

print(f"Skipped files: {skipped_files}")
print(f"Split into {len(all_chunks)} chunks")


# Skip existing ones
c.execute("SELECT hash FROM chunks")
existing_hashes = {row[0] for row in c.fetchall()}

new_chunks = [
    ch for ch in all_chunks
    if ch["hash"] not in existing_hashes
]

print(f"New chunks to index: {len(new_chunks)}")


# Process new chunks
for start in range(0, len(new_chunks), BATCH_SIZE):

    batch = new_chunks[start:start + BATCH_SIZE]

    texts = [b["text"] for b in batch]

    embeddings = emb_model.embed_documents(texts)

    rows = []

    for i, emb in enumerate(embeddings):
        rows.append((
            batch[i]["text"],
            batch[i]["source"],
            json.dumps(emb),
            batch[i]["hash"]
        ))

    c.executemany(
        """
        INSERT INTO chunks
        (text, source, embedding, hash)
        VALUES (?, ?, ?, ?)
        """,
        rows
    )

    conn.commit()

    print(
        f"Indexed {start + len(batch)} / {len(new_chunks)}"
    )

conn.close()

print("Finished! Vector DB ready.")