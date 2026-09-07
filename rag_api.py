import os
import json
import sqlite3
import time
import re
from typing import List, Dict, Optional, Tuple

import numpy as np
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings

DB_FILE = os.getenv("DB_FILE", "chunks.db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))

MAX_VECTORS = int(os.getenv("MAX_VECTORS", "0"))  # 0 = load all
FETCH_BATCH = int(os.getenv("FETCH_BATCH", "1000"))
LOG_EVERY = int(os.getenv("LOG_EVERY", "5000"))

# Groq model
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "700"))

# --- Retrieval tuning ---
MIN_SIM_THRESHOLD = float(os.getenv("MIN_SIM_THRESHOLD", "0.35"))  # κάτω από αυτό => keyword fallback
MIN_TEXT_LEN = int(os.getenv("MIN_TEXT_LEN", "120"))  # πετάμε πολύ μικρά/άχρηστα chunks

def log(msg: str):
    print(msg, flush=True)

app = Flask(__name__)

log("Initializing embedding model...")
t0 = time.time()
emb_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
log(f" Embedding model loaded: {EMBED_MODEL} (took {time.time()-t0:.1f}s)")

TEXTS: List[str] = []
SOURCES: List[str] = []
EMB: Optional[np.ndarray] = None

def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

def load_index_into_memory() -> None:
    global TEXTS, SOURCES, EMB

    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Δεν βρέθηκε το {DB_FILE}. Τρέξε πρώτα build_index.py.")

    limit = MAX_VECTORS if MAX_VECTORS else None

    log(f" Opening DB: {DB_FILE}")
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    total = 0
    try:
        if limit:
            c.execute("SELECT COUNT(*) FROM (SELECT 1 FROM chunks WHERE embedding IS NOT NULL LIMIT ?)", (limit,))
        else:
            c.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
        total = int(c.fetchone()[0] or 0)
    except Exception:
        total = 0

    if total:
        log(f" Rows to load: {total}")
    else:
        log("ℹ Counting rows failed (will still try to load).")

    if limit:
        c.execute("SELECT text, source, embedding FROM chunks WHERE embedding IS NOT NULL LIMIT ?", (limit,))
    else:
        c.execute("SELECT text, source, embedding FROM chunks WHERE embedding IS NOT NULL")

    texts: List[str] = []
    sources: List[str] = []
    embs: List[List[float]] = []

    loaded = 0
    t_read = time.time()

    while True:
        rows = c.fetchmany(FETCH_BATCH)
        if not rows:
            break
        for t, s, e in rows:
            if not t or not e:
                continue
            t = t.strip()
            if len(t) < MIN_TEXT_LEN:
                continue
            try:
                vec = json.loads(e)
            except Exception:
                continue

            texts.append(t)
            sources.append((s or "").strip())
            embs.append(vec)
            loaded += 1

            if LOG_EVERY and loaded % LOG_EVERY == 0:
                if total:
                    log(f"… loaded {loaded}/{total} rows (elapsed {time.time()-t_read:.1f}s)")
                else:
                    log(f"… loaded {loaded} rows (elapsed {time.time()-t_read:.1f}s)")

    conn.close()

    if loaded == 0:
        raise RuntimeError("Το DB δεν έχει embeddings ή όλα τα chunks είναι πολύ μικρά μετά το filtering.")

    log(f" Building numpy matrix from {loaded} vectors...")
    emb = np.array(embs, dtype=np.float32)
    emb = _normalize(emb)

    TEXTS, SOURCES, EMB = texts, sources, emb
    log(f" Loaded {loaded} vectors into memory")

def retrieve_semantic(query: str, top_k: int = TOP_K) -> List[Dict]:
    assert EMB is not None, "Index not loaded"

    q = np.array(
        emb_model.embed_query(query),
        dtype=np.float32
    ).reshape(1, -1)

    q = _normalize(q)

    sims = (EMB @ q.T).reshape(-1)

    k = min(top_k, len(sims))
    if k == 0:
        return []

    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    print("\n================ SEMANTIC RESULTS ================")

    results = []

    for pos, i in enumerate(idx.tolist(), start=1):

        print(f"{pos}. SCORE = {sims[i]:.3f}")
        print(f"SOURCE = {SOURCES[i]}")
        print(TEXTS[i][:250])
        print("-----------------------------------------------")

        results.append({
            "score": float(sims[i]),
            "text": TEXTS[i],
            "source": SOURCES[i]
        })

    print("=================================================\n")

    return results

def keyword_fallback(query: str, top_k: int = TOP_K) -> List[Dict]:

    print("\n================ KEYWORD FALLBACK ================\n")

    if not os.path.exists(DB_FILE):
        print("DB NOT FOUND:", DB_FILE)
        return []

    print("DB:", DB_FILE)
    print("QUESTION:", query)

    words = re.findall(
        r"[A-Za-zΑ-Ωα-ωΆΈΉΊΌΎΏάέήίόύώ0-9]{3,}",
        query
    )

    print("WORDS:", words)

    if not words:
        print("NO WORDS")
        return []

    words = words[:5]

    like_clauses = " AND ".join(["text LIKE ?"] * len(words))
    params = [f"%{w}%" for w in words]

    sql = f"""
    SELECT text, source
    FROM chunks
    WHERE {like_clauses}
    LIMIT ?
    """

    print("\nSQL:")
    print(sql)

    print("\nPARAMS:")
    print(params)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    try:
        c.execute(sql, (*params, top_k))
        rows = c.fetchall()

        print("\nFOUND:", len(rows))

        for i, (t, s) in enumerate(rows):
            print(f"\n----- ROW {i+1} -----")
            print("SOURCE:", s)
            print("TEXT:", t[:250])

    except Exception as e:
        print("\nSQL ERROR:")
        print(e)
        rows = []

    conn.close()

    hits = []

    for t, s in rows:
        if len((t or "").strip()) < MIN_TEXT_LEN:
            continue

        hits.append({
            "score": 1.0,
            "text": t,
            "source": s
        })

    print("\nRETURNING:", len(hits))
    print("=============================================\n")

    return hits

def build_prompt(context: str, question: str) -> str:
    return (
        "Είσαι ακαδημαϊκός βοηθός για πληροφορίες του Τμήματος.\n"
        "Απάντησε στα Ελληνικά.\n"
        "Χρησιμοποίησε ΜΟΝΟ τις πληροφορίες από το CONTEXT.\n"
        "Αν δεν υπάρχει απάντηση στο CONTEXT, πες ξεκάθαρα ότι δεν βρέθηκε.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ΑΠΑΝΤΗΣΗ:"
    )

def try_llm_answer(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None, "GROQ_API_KEY δεν είναι ορισμένο."

    try:
        from langchain_groq import ChatGroq
    except Exception as e:
        return None, f"No module 'langchain_groq' ({e}). Βάλε: pip install -U langchain-groq"

    try:
        llm = ChatGroq(api_key=api_key, model=GROQ_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp)), None
    except Exception as e:
        return None, str(e)

@app.route("/health", methods=["GET"])
def health():
    ok = EMB is not None and len(TEXTS) > 0
    return jsonify({
        "ok": ok,
        "rows_loaded": len(TEXTS),
        "embed_model": EMBED_MODEL,
        "llm_model": GROQ_MODEL,
        "max_vectors": MAX_VECTORS or None,
        "min_sim_threshold": MIN_SIM_THRESHOLD,
        "min_text_len": MIN_TEXT_LEN
    }), 200 if ok else 503

@app.route("/ask", methods=["POST"])
def ask():
    print("=" * 60)
    print("ASK CALLED")
    print("=" * 60, flush=True)
    data = request.get_json(silent=True) or {}
    print("RAW DATA:", data)
    print("QUESTION REPR:", repr(data.get("question")))
    print("QUESTION:", data.get("question"))
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    k = int(data.get("top_k") or TOP_K)

    # 1) semantic retrieval
    hits = retrieve_semantic(question, top_k=k)
    print("SEMANTIC HITS:", len(hits))
    print("BEST SCORE:", hits[0]["score"] if hits else None)
    

    # αν το καλύτερο score είναι χαμηλό => keyword fallback
    best_score = hits[0]["score"] if hits else 0.0
    if (not hits) or (best_score < MIN_SIM_THRESHOLD):
        print("USING KEYWORD FALLBACK")
        log(f" Low semantic score ({best_score:.3f}) -> keyword fallback")
        hits = keyword_fallback(question, top_k=k)
        print("KEYWORD HITS:", len(hits))
        
        
    if not hits:
        return jsonify({"answer": "Δεν βρέθηκε.", "sources": []})
    

    context = "\n\n---\n\n".join([h["text"] for h in hits])
    print("=" * 80)
    print("CONTEXT SENT TO LLM")
    print("=" * 80)
    print(context)
    print("=" * 80)

    prompt = build_prompt(context=context, question=question)

    answer, err = try_llm_answer(prompt)
    if answer is None:
        # fallback: επιστρέφουμε context + sources
        answer = "Δεν βρέθηκε." if "δεν βρέθηκε" in context.lower() else context

    # unique sources
    unique_sources, seen = [], set()
    for h in hits:
        src = (h.get("source") or "").strip()
        if src and src not in seen:
            seen.add(src)
            unique_sources.append(src)

    return jsonify({"answer": answer, "sources": unique_sources, "top_k": k, "best_score": best_score})

if __name__ == "__main__":
    load_index_into_memory()
    log(" Starting RAG API on http://127.0.0.1:5000 ...")
    # debug=False για να ΜΗΝ ξαναφορτώνει 2 φορές
    app.run(host="0.0.0.0", port=5000, debug=False)
