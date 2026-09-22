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

## Final V2 tests

### 1. Faculty query that previously failed

**Question:** `Ποια μαθήματα διδάσκει ο κ. Μενύχτας;`

**Result:** PASS

V2 returned six courses:
- Γλώσσα Προγραμματισμού C — 1ο εξάμηνο
- Αντικειμενοστρεφής Προγραμματισμός — 2ο εξάμηνο
- Υπολογιστικά Νέφη — 5ο εξάμηνο
- Διαδικτυακά Πληροφοριακά Συστήματα — 6ο εξάμηνο
- Υπηρεσιοστρεφείς Αρχιτεκτονικές και Φορητή Υπολογιστική — 7ο εξάμηνο
- Μεθοδολογίες Ανάπτυξης Πληροφοριακών Συστημάτων — 8ο εξάμηνο

### 2. Generalization — Φιλιππάκης

**Question:** `Ποια μαθήματα διδάσκει ο κ. Φιλιππάκης;`

**Result:** PASS

The system returned six courses associated with the faculty member.

### 3. Generalization — Μηλιώνης

**Question:** `Ποια μαθήματα διδάσκει ο κ. Μηλιώνης;`

**Result:** PASS

The system returned five courses associated with the faculty member.

### 4. Semester regression test

**Question:** `Ποια είναι τα μαθήματα του 5ου εξαμήνου;`

**Result:** PASS

The system returned the complete 13-course list available in the structured semester information, including course codes and categories.

### 5. End-to-end Chainlit test

The Menychtas query was also executed through the V2 Chainlit interface.

**Result:** PASS

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
