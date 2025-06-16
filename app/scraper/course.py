"""
app/scraper/course.py
───────────────────────────────────────────────────────────────────────────────
Download every Markdown page in the "tools-in-data-science-public" repo,
convert it to plain text, and save data/course.json for your RAG pipeline.

• Uses the GitHub REST API (no auth needed, but PAT recommended)
• Recursively walks the repo tree, picking *.md files (skips images, flowcharts…)
• Extracts a decent title (first H1 or filename)
• Converts Markdown → plain text with markdown-it-py + BeautifulSoup
"""

from __future__ import annotations
import base64, json, os, pathlib, re, textwrap
from urllib.parse import urljoin

import requests, tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from markdown_it import MarkdownIt

# ───────────────────────── Config ─────────────────────────────────────────────
load_dotenv()
OWNER  = "sanand0"
REPO   = "tools-in-data-science-public"
BRANCH = "main"
API    = f"https://api.github.com/repos/{OWNER}/{REPO}"
RAW    = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}"
OUT    = pathlib.Path("data/course.json")
MD     = MarkdownIt()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # optional but avoids 60‑requests/hour cap
HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# ──────────────────────── Helpers ─────────────────────────────────────────────
def github_get(endpoint: str):
    r = requests.get(urljoin(API + "/", endpoint.lstrip("/")), headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def walk_tree(sha: str, prefix="") -> list[str]:
    """Recursively collect Markdown file paths from a git tree SHA."""
    tree = github_get(f"git/trees/{sha}").get("tree", [])
    md_files = []
    for item in tree:
        path = f"{prefix}/{item['path']}".lstrip("/")
        if item["type"] == "tree":
            md_files += walk_tree(item["sha"], prefix=path)
        elif item["type"] == "blob" and path.lower().endswith(".md"):
            md_files.append(path)
    return md_files

def md_to_text(markdown: str) -> str:
    html = MD.render(markdown)
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)

TITLE_RE = re.compile(r"^#\s+(.+?)$", re.MULTILINE)
def extract_title(md_text: str, fallback: str) -> str:
    m = TITLE_RE.search(md_text)
    return m.group(1).strip() if m else fallback.replace("-", " ").title()

def slug_from_path(path: str) -> str:
    return pathlib.Path(path).stem.lower().replace(" ", "-")

# ──────────────────────── Main scraper ────────────────────────────────────────
def main():
    print("🐙  Fetching latest commit SHA …")
    branch_info = github_get(f"branches/{BRANCH}")
    root_sha = branch_info["commit"]["commit"]["tree"]["sha"]

    print("📂  Walking repository tree …")
    md_paths = walk_tree(root_sha)
    print(f"➡️   Found {len(md_paths)} Markdown files")

    corpus = []
    for path in tqdm.tqdm(md_paths, desc="Downloading", unit="file"):
        raw_url = f"{RAW}/{path}"
        r = requests.get(raw_url, timeout=30)
        if r.status_code != 200:
            continue
        md_text = r.text
        title = extract_title(md_text, pathlib.Path(path).stem)
        text = textwrap.shorten(md_to_text(md_text), width=10_000, placeholder=" …")
        corpus.append(
            {
                "id": slug_from_path(path),
                "url": raw_url,
                "title": title,
                "text": text,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    print(f"\n✅  Saved {len(corpus)} pages → {OUT.resolve()}")

if __name__ == "__main__":
    main()
