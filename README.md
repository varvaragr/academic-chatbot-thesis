# Ψηφιακός Βοηθός Φοιτητή — Τμήμα Ψηφιακών Συστημάτων

RAG chatbot που απαντά σε ερωτήσεις με βάση αποθηκευμένο περιεχόμενο από τον ιστότοπο του Τμήματος Ψηφιακών Συστημάτων του Πανεπιστημίου Πειραιώς.

Η τελική έκδοση χρησιμοποιεί υβριδική ανάκτηση πληροφορίας (semantic + lexical retrieval), συνδυασμό αποτελεσμάτων με Reciprocal Rank Fusion (RRF) και το `openai/gpt-oss-120b` μέσω Groq για την παραγωγή της τελικής απάντησης.

## Βασικά αρχεία

- `scraper.py`: συλλογή περιεχομένου από τον ιστότοπο.
- `ds_content.json`: αποθηκευμένο corpus.
- `build_index.py`: τεμαχισμός κειμένου και δημιουργία embeddings/SQLite index.
- `rag_api.py`: Flask RAG API, hybrid retrieval/RRF και σύνδεση με Groq.
- `chainlit_app.py`: διεπαφή συνομιλίας Chainlit.
- `docker-compose.yml`: εκκίνηση API και UI.
- `evaluation/`: scripts και αποτελέσματα της τελικής αξιολόγησης.

## Εκτέλεση με Docker

### 1. Δημιουργία `.env`

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Άνοιξε το `.env` και αντικατάστησε:

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY_HERE
```

με το προσωπικό Groq API key. Το `.env` δεν πρέπει να γίνει commit στο Git.

### 2. Build και εκκίνηση

```bash
docker compose build
docker compose up
```

Στην πρώτη εκτέλεση γίνεται λήψη του embedding model και, αν δεν υπάρχει ήδη η βάση στο Docker volume, δημιουργείται το `chunks.db` από το `ds_content.json`. Έπειτα ξεκινούν το Flask API και το Chainlit UI.

Η πρώτη εκτέλεση μπορεί να χρειαστεί αρκετά λεπτά.

### 3. Διευθύνσεις

- Chainlit: `http://localhost:8000`
- API health: `http://localhost:5000/health`

### 4. Τερματισμός

```bash
docker compose down
```

Για πλήρη διαγραφή της Docker βάσης και αναδημιουργία της στο επόμενο `up`:

```bash
docker compose down -v
```

## Τοπική εκτέλεση χωρίς Docker

Δημιουργία virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Εγκατάσταση εξαρτήσεων:

```bash
pip install -r requirements.txt
```

Ορισμός του `GROQ_API_KEY` στο environment ή μέσω του `.env`, ανάλογα με τον τρόπο εκτέλεσης.

Δημιουργία βάσης:

```bash
python build_index.py
```

Εκκίνηση API:

```bash
python rag_api.py
```

Σε δεύτερο terminal:

```bash
chainlit run chainlit_app.py --port 8000
```

## Αξιολόγηση

Η τελική αξιολόγηση βασίστηκε σε 20 ερωτήματα, οργανωμένα σε πέντε κατηγορίες: πραγματολογικά, παραφρασμένα, ερωτήματα εξειδικευμένων όρων, σύνθετα ερωτήματα και ερωτήματα ανεπαρκούς πληροφορίας.

Ο φάκελος `evaluation/` περιλαμβάνει:

- `rag_api_evaluation.py`: έκδοση του API που επιστρέφει και τα ανακτημένα contexts για τις ανάγκες της αξιολόγησης.
- `evaluate_chatbot_with_contexts.py`: εκτέλεση των 20 ερωτημάτων και δημιουργία του `evaluation_results.json`.
- `evaluate_ragas_llm_only_fixed.py`: αξιολόγηση με RAGAS για Faithfulness, Context Precision with Reference και Context Recall.
- `ragas_results_llm_only.json`: αναλυτικά αποτελέσματα RAGAS.
- `ragas_summary_llm_only.json`: συνοπτικά αποτελέσματα RAGAS.
- `evaluate_answer_relevancy.py`: αυτοτελής αξιολόγηση Answer Relevancy με plain-text LLM-as-a-judge, ακολουθώντας τη λογική της αντίστοιχης μετρικής του DeepEval χωρίς structured output/tool calling.
- `deepeval_results_final.json`: τελικά αποτελέσματα Answer Relevancy του evaluation run που χρησιμοποιήθηκε στην πτυχιακή.

### Ενδεικτική σειρά επανεκτέλεσης

Με ενεργό το evaluation API:

```bash
python evaluation/rag_api_evaluation.py
```

Σε δεύτερο terminal:

```bash
python evaluation/evaluate_chatbot_with_contexts.py
```

Για RAGAS:

```bash
python evaluation/evaluate_ragas_llm_only_fixed.py
```

Για Answer Relevancy:

```bash
python evaluation/evaluate_answer_relevancy.py
```

> Σημείωση: τα αποθηκευμένα αποτελέσματα στο repository προέρχονται από το τελικό evaluation run της εργασίας. Νέα εκτέλεση με LLM-as-a-judge μπορεί να παρουσιάσει μικρές αποκλίσεις λόγω της φύσης των γλωσσικών μοντέλων και της εξωτερικής υπηρεσίας.

## Git

Δεν ανεβαίνουν στο repository τα `.env`, `chunks.db`, virtual environments, caches ή logs. Η βάση ανακατασκευάζεται από το `ds_content.json` με το `build_index.py`.
