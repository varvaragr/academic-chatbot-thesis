import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag
import json
import re
from collections import deque


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://www.ds.unipi.gr"
OUTPUT_FILE = "ds_content.json"


SKIP_EXTENSIONS = (
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".ico",
    ".zip",
    ".rar",
    ".7z",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".mp3",
    ".wav",
    ".mp4",
    ".avi",
    ".mov"
)


SKIP_PATTERNS = [
    "/feed/",
    "/comments/",
    "/wp-json/",
    "/xmlrpc.php",
    "/wp-admin/",
]


# ============================================================
# SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
})


# ============================================================
# HELPERS
# ============================================================

def normalize_url(url):

    url, _ = urldefrag(url)

    if url.endswith("/"):
        url = url[:-1]

    return url



def should_skip(url):

    url = url.lower()

    if not url.startswith(BASE_URL):
        return True


    if any(url.endswith(ext) for ext in SKIP_EXTENSIONS):
        return True


    if any(pattern in url for pattern in SKIP_PATTERNS):
        return True


    return False



def extract_text(soup):

    # Αφαιρούμε άχρηστα στοιχεία

    for tag in soup([
        "script",
        "style",
        "noscript",
        "header",
        "footer",
        "nav",
        "aside",
        "form"
    ]):

        tag.decompose()



    # WordPress κύριο περιεχόμενο

    content = soup.select_one(
        "div.entry-content"
    )


    if content is None:
        content = soup.find("article")


    if content is None:
        content = soup.find("main")


    if content is None:
        content = soup.body



    if content is None:
        return ""



    text = content.get_text(
        separator=" ",
        strip=True
    )


    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()



def extract_links(soup, current_url):

    links = []


    for a in soup.find_all("a", href=True):

        href = a["href"]


        full_url = urljoin(
            current_url,
            href
        )


        full_url = normalize_url(
            full_url
        )


        if not should_skip(full_url):

            links.append(full_url)



    return links



# ============================================================
# SCRAPER
# ============================================================

def crawl():

    visited = set()

    queue = deque()

    queue.append(BASE_URL)


    documents = []


    while queue:


        url = queue.popleft()


        url = normalize_url(url)


        if url in visited:
            continue


        if should_skip(url):
            continue



        visited.add(url)


        print("=" * 60)
        print("SCRAPING:")
        print(url)



        try:

            response = session.get(
                url,
                timeout=20
            )


        except Exception as e:

            print("ERROR:", e)
            continue



        if response.status_code != 200:

            print(
                "STATUS:",
                response.status_code
            )

            continue



        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )


        text = extract_text(
            soup
        )



        if len(text) > 100:


            documents.append({

                "url": url,

                "text": text

            })


            print(
                "Saved"
            )


        else:

            print(
                "Skipped - small content"
            )



        # Βρίσκουμε νέα links

        for link in extract_links(
            soup,
            url
        ):


            if link not in visited:

                queue.append(link)



    return documents



# ============================================================
# SAVE
# ============================================================

def save_data(data):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:


        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":


    print("\n")
    print("=" * 60)
    print("START SCRAPING")
    print(BASE_URL)
    print("=" * 60)



    pages = crawl()



    save_data(
        pages
    )



    print("\n")
    print("=" * 60)
    print("FINISHED")
    print(
        "TOTAL PAGES:",
        len(pages)
    )
    print(
        "Saved:",
        OUTPUT_FILE
    )
    print("=" * 60)