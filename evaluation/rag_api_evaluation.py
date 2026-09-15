import os
import json
import sqlite3
import time
import re
import unicodedata
from typing import List, Dict, Optional, Tuple

import numpy as np
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings

DB_FILE = os.getenv("DB_FILE", "chunks.db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_VECTORS = int(os.getenv("MAX_VECTORS", "0"))
FETCH_BATCH = int(os.getenv("FETCH_BATCH", "1000"))
LOG_EVERY = int(os.getenv("LOG_EVERY", "5000"))

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "700"))
MIN_TEXT_LEN = int(os.getenv("MIN_TEXT_LEN", "120"))

SEMANTIC_CANDIDATES = int(os.getenv("SEMANTIC_CANDIDATES", "20"))
KEYWORD_CANDIDATES = int(os.getenv("KEYWORD_CANDIDATES", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "1.0"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "1.8"))


def log(msg: str):
    print(msg, flush=True)


app = Flask(__name__)

log("Initializing embedding model...")
t0 = time.time()
emb_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
log(f"Embedding model loaded: {EMBED_MODEL} (took {time.time()-t0:.1f}s)")

TEXTS: List[str] = []
SOURCES: List[str] = []
EMB: Optional[np.ndarray] = None


def _normalize_vectors(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def load_index_into_memory() -> None:
    global TEXTS, SOURCES, EMB

    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Δεν βρέθηκε το {DB_FILE}. Τρέξε πρώτα build_index.py.")

    limit = MAX_VECTORS if MAX_VECTORS else None
    log(f"Opening DB: {DB_FILE}")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    try:
        if limit:
            c.execute("SELECT COUNT(*) FROM (SELECT 1 FROM chunks WHERE embedding IS NOT NULL LIMIT ?)", (limit,))
        else:
            c.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
        total = int(c.fetchone()[0] or 0)
    except Exception:
        total = 0

    if total:
        log(f"Rows to load: {total}")

    if limit:
        c.execute("SELECT text, source, embedding FROM chunks WHERE embedding IS NOT NULL LIMIT ?", (limit,))
    else:
        c.execute("SELECT text, source, embedding FROM chunks WHERE embedding IS NOT NULL")

    texts, sources, embs = [], [], []
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
                    log(f"... loaded {loaded}/{total} rows (elapsed {time.time()-t_read:.1f}s)")
                else:
                    log(f"... loaded {loaded} rows (elapsed {time.time()-t_read:.1f}s)")

    conn.close()

    if loaded == 0:
        raise RuntimeError("Το DB δεν έχει embeddings ή όλα τα chunks είναι πολύ μικρά μετά το filtering.")

    log(f"Building numpy matrix from {loaded} vectors...")
    emb = np.array(embs, dtype=np.float32)
    emb = _normalize_vectors(emb)

    TEXTS, SOURCES, EMB = texts, sources, emb
    log(f"Loaded {loaded} vectors into memory")


def retrieve_semantic(query: str, top_k: int = SEMANTIC_CANDIDATES) -> List[Dict]:
    assert EMB is not None, "Index not loaded"

    if len(EMB) == 0:
        return []

    q = np.array(emb_model.embed_query(query), dtype=np.float32).reshape(1, -1)
    q = _normalize_vectors(q)
    sims = (EMB @ q.T).reshape(-1)

    k = min(max(1, top_k), len(sims))
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
            "source": SOURCES[i],
            "retrieval": "semantic"
        })

    print("=================================================\n")
    return results


GREEK_STOPWORDS = {
    "και", "για", "απο", "στο", "στη", "στην", "στις", "στους", "των", "τον", "την",
    "του", "της", "τα", "το", "οι", "η", "ο", "σε", "με", "να", "που", "πως", "ποιο",
    "ποια", "ποιες", "ποιος", "ποιοι", "τι", "ειναι", "εχει", "εχουν", "υπαρχει", "υπαρχουν",
    "μπορω", "βρω", "βρισκεται", "βρισκονται", "δωσε", "πες", "μου", "θα", "ως", "ενα", "μια",
    "ενος", "μιας", "αυτο", "αυτη", "αυτα", "σχετικα"
}

ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "what",
    "where", "how", "which", "who", "with", "about", "me", "tell", "can", "i"
}


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^0-9a-zα-ω]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def query_tokens(query: str) -> List[str]:
    tokens = re.findall(r"[0-9a-zα-ω]+", normalize_text(query))
    useful = []

    for token in tokens:
        if len(token) < 3:
            continue
        if token in GREEK_STOPWORDS or token in ENGLISH_STOPWORDS:
            continue
        if token not in useful:
            useful.append(token)

    return useful


def retrieve_keyword(query: str, top_k: int = KEYWORD_CANDIDATES) -> List[Dict]:
    tokens = query_tokens(query)

    print("\n================ KEYWORD RESULTS =================")
    print("TOKENS:", tokens)

    if not tokens:
        print("No useful keyword tokens.")
        print("=================================================\n")
        return []

    scored = []

    for i, text in enumerate(TEXTS):
        normalized_text = normalize_text(text)
        matched = [token for token in tokens if token in normalized_text]

        if not matched:
            continue

        coverage = len(matched) / len(tokens)
        match_bonus = min(len(matched) * 0.05, 0.25)
        score = coverage + match_bonus
        scored.append((score, len(matched), i, matched))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    results = []

    for pos, (score, matched_count, i, matched) in enumerate(scored[:top_k], start=1):
        print(f"{pos}. SCORE = {score:.3f}")
        print(f"MATCHED = {matched}")
        print(f"SOURCE = {SOURCES[i]}")
        print(TEXTS[i][:250])
        print("-----------------------------------------------")

        results.append({
            "score": float(score),
            "text": TEXTS[i],
            "source": SOURCES[i],
            "retrieval": "keyword",
            "matched_terms": matched
        })

    print("=================================================\n")
    return results


def _hit_key(hit: Dict) -> Tuple[str, str]:
    return ((hit.get("source") or "").strip(), (hit.get("text") or "").strip())


def hybrid_retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    candidate_k = max(top_k * 4, SEMANTIC_CANDIDATES, KEYWORD_CANDIDATES)

    semantic_hits = retrieve_semantic(query, top_k=candidate_k)
    keyword_hits = retrieve_keyword(query, top_k=candidate_k)

    fused: Dict[Tuple[str, str], Dict] = {}

    for rank, hit in enumerate(semantic_hits, start=1):
        key = _hit_key(hit)
        if key not in fused:
            fused[key] = {
                "text": hit["text"],
                "source": hit["source"],
                "rrf_score": 0.0,
                "semantic_score": None,
                "keyword_score": None,
                "retrieval_methods": []
            }

        fused[key]["rrf_score"] += SEMANTIC_WEIGHT / (RRF_K + rank)
        fused[key]["semantic_score"] = hit["score"]
        if "semantic" not in fused[key]["retrieval_methods"]:
            fused[key]["retrieval_methods"].append("semantic")

    for rank, hit in enumerate(keyword_hits, start=1):
        key = _hit_key(hit)
        if key not in fused:
            fused[key] = {
                "text": hit["text"],
                "source": hit["source"],
                "rrf_score": 0.0,
                "semantic_score": None,
                "keyword_score": None,
                "retrieval_methods": []
            }

        fused[key]["rrf_score"] += KEYWORD_WEIGHT / (RRF_K + rank)
        fused[key]["keyword_score"] = hit["score"]
        if "keyword" not in fused[key]["retrieval_methods"]:
            fused[key]["retrieval_methods"].append("keyword")

    ranked = sorted(fused.values(), key=lambda h: h["rrf_score"], reverse=True)
    final_hits = ranked[:top_k]

    print("\n================ HYBRID FINAL RESULTS ============")

    for pos, hit in enumerate(final_hits, start=1):
        methods = "+".join(hit["retrieval_methods"])
        print(f"{pos}. RRF = {hit['rrf_score']:.6f}")
        print(f"METHOD = {methods}")
        print(f"SEMANTIC = {hit['semantic_score']}")
        print(f"KEYWORD = {hit['keyword_score']}")
        print(f"SOURCE = {hit['source']}")
        print(hit["text"][:250])
        print("-----------------------------------------------")

    print("=================================================\n")
    return final_hits


def build_prompt(context: str, question: str) -> str:
    return (
        "Είσαι ακαδημαϊκός βοηθός για πληροφορίες του Τμήματος Ψηφιακών Συστημάτων του Πανεπιστημίου Πειραιώς.\n"
        "Απάντησε στα Ελληνικά.\n"
        "Χρησιμοποίησε ΜΟΝΟ τις πληροφορίες από το CONTEXT.\n"
        "Μην επινοείς πληροφορίες και μην χρησιμοποιείς εξωτερικές γνώσεις.\n"
        "Αν δεν υπάρχει επαρκής απάντηση στο CONTEXT, πες ξεκάθαρα ότι δεν βρέθηκε σχετική πληροφορία.\n\n"
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
        llm = ChatGroq(
            api_key=api_key,
            model=GROQ_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
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
        "min_text_len": MIN_TEXT_LEN,
        "retrieval": "hybrid_rrf",
        "semantic_weight": SEMANTIC_WEIGHT,
        "keyword_weight": KEYWORD_WEIGHT
    })


@app.route("/ask", methods=["POST"])
def ask():
    print("=" * 60)
    print("ASK CALLED")
    print("=" * 60, flush=True)

    data = request.get_json(silent=True) or {}
    print("RAW DATA:", data)

    question = (data.get("question") or "").strip()
    print("QUESTION:", question)

    if not question:
        return jsonify({"error": "No question provided"}), 400

    k = int(data.get("top_k") or TOP_K)
    hits = hybrid_retrieve(question, top_k=k)

    if not hits:
        return jsonify({
            "answer": "Δεν βρέθηκε σχετική πληροφορία.",
            "sources": [],
            "top_k": k,
            "best_semantic_score": None,
            "retrieval": "hybrid_rrf",
            "retrieved_contexts": []
        })

    context = "\n\n---\n\n".join(h["text"] for h in hits)

    print("=" * 80)
    print("CONTEXT SENT TO LLM")
    print("=" * 80)
    print(context)
    print("=" * 80)

    prompt = build_prompt(context=context, question=question)
    answer, err = try_llm_answer(prompt)

    if answer is None:
        log(f"LLM ERROR: {err}")
        answer = "Δεν ήταν δυνατή η παραγωγή απάντησης από το γλωσσικό μοντέλο."

    unique_sources, seen = [], set()

    for hit in hits:
        src = (hit.get("source") or "").strip()
        if src and src not in seen:
            seen.add(src)
            unique_sources.append(src)

    semantic_scores = [
        h.get("semantic_score")
        for h in hits
        if h.get("semantic_score") is not None
    ]
    best_semantic_score = max(semantic_scores) if semantic_scores else None

    return jsonify({
        "answer": answer,
        "sources": unique_sources,
        "top_k": k,
        "best_semantic_score": best_semantic_score,
        "retrieval": "hybrid_rrf",
        "retrieved_contexts": [h["text"] for h in hits]
    })


if __name__ == "__main__":
    load_index_into_memory()
    log("Starting RAG API on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
