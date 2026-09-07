# Ψηφιακός Βοηθός Φοιτητή — Τμήμα Ψηφιακών Συστημάτων

RAG chatbot που απαντά σε ερωτήσεις με βάση αποθηκευμένο περιεχόμενο από τον ιστότοπο του Τμήματος Ψηφιακών Συστημάτων του Πανεπιστημίου Πειραιώς.

## Βασικά αρχεία

- `scrapper.py`: συλλογή περιεχομένου από τον ιστότοπο.
- `ds_content.json`: αποθηκευμένο corpus.
- `build_index.py`: τεμαχισμός κειμένου και δημιουργία embeddings/SQLite index.
- `rag_api.py`: Flask RAG API και σύνδεση με Groq.
- `chainlit_app.py`: διεπαφή συνομιλίας Chainlit.
- `docker-compose.yml`: εκκίνηση API και UI.

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

Στην πρώτη εκτέλεση γίνεται λήψη του embedding model και, αν δεν υπάρχει ήδη η βάση στο Docker volume, δημιουργείται αυτόματα το `chunks.db` από το `ds_content.json`. Έπειτα ξεκινούν το Flask API και το Chainlit UI.

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

Εγκατάσταση:

```bash
pip install -r requirements.txt
```

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

## Git

Δεν ανεβαίνουν στο repository τα `.env`, `chunks.db`, virtual environments, caches ή logs. Η βάση ανακατασκευάζεται από το `ds_content.json` με το `build_index.py`.
