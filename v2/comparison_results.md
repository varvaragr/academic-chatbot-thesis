# Original vs V2 — Validation Results

## Scope

V2 was developed after evaluation of the original academic chatbot. The goal was not to hard-code answers to individual questions, but to improve retrieval and knowledge representation so that the solution also generalizes to similar queries.

## Main changes

| Component | Original | V2 |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` | `Alibaba-NLP/gte-multilingual-base` |
| Knowledge base | Original website corpus | Expanded and structured corpus |
| Faculty-course information | General scraped text | Structured faculty-course records |
| Retrieval | Hybrid semantic + lexical + RRF | Hybrid semantic + IDF-aware lexical + RRF, deduplication and semantic preservation |
| Top-K | 5 | 5 |
| LLM | `openai/gpt-oss-120b` | `openai/gpt-oss-120b` |
| Generation budget | 700 tokens | 2000 tokens |


```text
Chainlit V2
    ↓
Flask RAG API V2
    ↓
V2 SQLite vector index
    ↓
Hybrid retrieval
    ↓
GPT-OSS through Groq
    ↓
Answer displayed in Chainlit
```

## Conclusion

The final tests show that V2 resolves the query patterns that motivated the revision and also produces successful results for additional faculty members. The original implementation remains separate and unchanged, allowing the baseline and V2 to be compared independently.
