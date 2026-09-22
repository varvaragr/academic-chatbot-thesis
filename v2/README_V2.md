# Academic Chatbot V2

Experimental second version of the academic chatbot for the Department of Digital Systems, University of Piraeus.

## Purpose

V2 was developed after testing the original system and observing retrieval limitations in questions that require complete structured information, such as courses taught by a faculty member or courses belonging to a specific semester.

## Main improvements

- Embedding model: `Alibaba-NLP/gte-multilingual-base`
- Expanded and better-structured knowledge corpus from official Department sources
- Structured faculty-course and semester information
- Hybrid semantic and IDF-aware lexical retrieval
- Reciprocal Rank Fusion (RRF)
- Near-duplicate filtering and preservation of strong semantic hits
- Top-K: 5
- LLM: `openai/gpt-oss-120b` through Groq
- Generation budget: 2000 tokens

## Files

- `build_knowledge_base.py` — creates structured faculty-course records.
- `base2.json` — expanded V2 knowledge corpus.
- `build_index2.py` — builds the V2 SQLite vector index.
- `rag_api2.py` — Flask RAG backend.
- `chainlit_app2.py` — Chainlit user interface.
- `comparison_results.md` — summary of final V2 validation tests.

The generated SQLite vector database is not included because it can be rebuilt locally.

## Installation

Python 3.10+ is recommended.

```bash
pip install flask chainlit httpx numpy langchain-huggingface langchain-groq sentence-transformers
```

Set the Groq API key. On Windows PowerShell:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Never commit API keys or `.env` files.

## Build the V2 index

```powershell
python build_index_v2_faculty.py
```

This creates `chunks_v2_faculty.db`.

## Run the backend

```powershell
python rag_api_v2_faculty.py
```

The V2 API uses port `5001` by default.

## Run the interface

In a second terminal:

```powershell
chainlit run chainlit_app_v2.py
```

## Validation

The final V2 was tested on the query types that motivated the revision and on additional faculty queries to check generalization. See `comparison_results.md`.

## Versioning note

V2 is kept separate from the original thesis implementation so the original baseline remains reproducible and the post-evaluation improvements can be examined independently.
