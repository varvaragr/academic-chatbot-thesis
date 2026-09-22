import json
import re
import unicodedata
from pathlib import Path

INPUT_FILE = Path("ds_content_v2_final.json")
OUTPUT_FILE = Path("ds_content_v2_faculty_aggregated.json")

END_MARKERS = [
    "Σύντομο βιογραφικό", "Βιογραφικό", "Ερευνητικά ενδιαφέροντα",
    "Ερευνητικό έργο", "Δημοσιεύσεις", "Εκπαίδευση", "Σπουδές"
]

def clean(text):
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip()

def base_faculty_url(url):
    return (url or "").split("#", 1)[0].rstrip("/")

def extract_name(text):
    # Faculty pages start with the member's full name before office/contact metadata.
    m = re.match(
        r"^\s*(.+?)\s+(?:Γραφείο|Τηλέφωνο|Email|E-mail|Σύνδεσμοι|Σύνδεσμος|Ιστοσελίδα|Τίτλος|Ώρες Γραφείου)\b",
        clean(text),
        re.I,
    )
    return clean(m.group(1)) if m else None

def extract_teaching(text):
    t = clean(text)
    marker = "Διδασκαλία"
    pos = t.find(marker)
    if pos < 0:
        return None
    teaching = t[pos + len(marker):].strip()
    cut = len(teaching)
    for end in END_MARKERS:
        p = teaching.find(end)
        if p >= 0:
            cut = min(cut, p)
    teaching = teaching[:cut].strip()
    return teaching if len(teaching) >= 3 else None

data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

# Prefer the original faculty page for each person; ignore prior #v2-structured
# duplicates while creating the new aggregation layer.
faculty_pages = {}
for doc in data:
    url = doc.get("url", "")
    if "/faculty/" not in url:
        continue
    base = base_faculty_url(url)
    if not base or base in faculty_pages:
        continue
    if "#v2-structured" not in url:
        faculty_pages[base] = doc

aggregated = []
for base, doc in faculty_pages.items():
    name = extract_name(doc.get("text", ""))
    teaching = extract_teaching(doc.get("text", ""))
    if not name or not teaching:
        continue

    text = (
        f"Διδάσκων/Διδάσκουσα: {name}\n"
        f"Διδασκαλία / Μαθήματα που διδάσκει:\n{teaching}\n"
        f"Πηγή: επίσημη σελίδα μέλους ΔΕΠ του Τμήματος Ψηφιακών Συστημάτων."
    )
    aggregated.append({
        "url": base + "#v2-faculty-courses",
        "text": text
    })

# Idempotent: remove records from an earlier run, then append freshly generated ones.
base_data = [
    d for d in data
    if "#v2-faculty-courses" not in d.get("url", "")
]
final_data = base_data + aggregated
OUTPUT_FILE.write_text(
    json.dumps(final_data, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print(f"Input records: {len(data)}")
print(f"Faculty-course records added: {len(aggregated)}")
print(f"Output records: {len(final_data)}")
print(f"Saved: {OUTPUT_FILE}")
