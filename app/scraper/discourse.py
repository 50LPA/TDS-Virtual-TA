"""
app/scraper/discourse.py
───────────────────────────────────────────────────────────────────────────────
Scrape IITM‑Online‑Degree Discourse posts (1 Jan – 14 Apr 2025) WITHOUT an
API key. First run opens a visible Chromium window so you can complete SSO.
Cookies are saved to .auth/state.json; subsequent runs are fully headless.

Output → data/discourse.json
"""

from __future__ import annotations
import asyncio, datetime as dt, json, os, pathlib, time
from urllib.parse import urljoin

import requests, tqdm
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# ──────────────────────────── Config ──────────────────────────────────────────
load_dotenv()

BASE_URL    = os.getenv("BASE_URL",   "https://discourse.onlinedegree.iitm.ac.in")
START_DATE  = dt.date.fromisoformat(os.getenv("FROM_DATE", "2025-01-01"))
END_DATE    = dt.date.fromisoformat(os.getenv("TO_DATE",   "2025-04-14"))
OUT_FILE    = pathlib.Path("data/discourse.json")
STATE_DIR   = pathlib.Path(".auth"); STATE_DIR.mkdir(exist_ok=True)
STATE_FILE  = STATE_DIR / "state.json"
LOGIN_WAIT  = 120          # seconds to wait for you to finish SSO on 1st run
RATE_DELAY  = 0.5          # polite delay between /posts.json requests (sec)

# ────────────────────────── Auth helpers ──────────────────────────────────────
async def ensure_storage_state() -> dict:
    """
    • If .auth/state.json exists ➜ return its dict.
    • Otherwise open headed Chromium, let user log in, save storage state,
      and return it.
    """
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())

    print(
        "\n🖥  First run: a browser will open.\n"
        "   • Sign in via IITM SSO as usual.\n"
        "   • When you reach the forum’s /latest page, close the window.\n"
    )
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, timeout=LOGIN_WAIT * 1000)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(urljoin(BASE_URL, "/login"), wait_until="domcontentloaded")

        try:
            await page.wait_for_url(f"{BASE_URL}/latest*", timeout=LOGIN_WAIT * 1000)
        except Exception:
            print(f"❌  Login not completed within {LOGIN_WAIT}s — exiting.")
            await browser.close()
            raise SystemExit(1)

        state = await context.storage_state()
        STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
        print(f"✅  Login cookies saved ➜ {STATE_FILE}")
        await browser.close()
        return state


def build_cookie_header(state: dict) -> str:
    """Return a 'Cookie:' header string containing all forum cookies."""
    return "; ".join(
        f"{ck['name']}={ck['value']}"
        for ck in state["cookies"]
        if ck["domain"].endswith("iitm.ac.in")
    )

# ────────────────────────── Scraper core ──────────────────────────────────────
def crawl_posts(cookie_header: str) -> list[dict]:
    """
    Harvest posts BETWEEN START_DATE and END_DATE using proper ‘before’
    pagination. Each /posts.json call returns up to 50 posts ordered
    NEWEST → OLDEST. We keep calling
        /posts.json?before=<lowest_id_seen_so_far>
    until the oldest post in a batch is earlier than START_DATE.
    """
    sess = requests.Session()
    sess.headers.update({
        "Cookie":     cookie_header,
        "User-Agent": "virtual-ta-scraper/1.0 (+https://github.com/you/tds-virtual-ta)"
    })

    harvested: list[dict] = []
    before: int | None = None          # post‑id cursor
    batch_no = 0

    with tqdm.tqdm(desc="Downloading pages", unit="page") as bar:
        while True:
            url = f"{BASE_URL}/posts.json"
            if before:
                url += f"?before={before}"

            resp = sess.get(url, timeout=30)
            try:
                data = resp.json()
                posts_raw = data.get("latest_posts") or data.get("latest")
            except Exception:
                print("\n❌  Unexpected response, aborting.\n", resp.text[:300])
                break

            # Empty response = no more posts
            if not posts_raw:
                break

            # Track progress
            batch_no += 1
            bar.update()

            # Oldest post in this batch
            oldest_post = min(posts_raw, key=lambda p: p["id"])
            before = oldest_post["id"]  # cursor for next call

            # Stop condition – once we see any post BEFORE our start date
            oldest_date = dt.date.fromisoformat(oldest_post["created_at"][:10])
            if oldest_date < START_DATE:
                # keep only those within range and exit
                posts_raw = [p for p in posts_raw
                             if START_DATE <= dt.date.fromisoformat(p["created_at"][:10]) <= END_DATE]
                harvested.extend(posts_raw)
                break

            # Normal case – filter & keep going
            for post in posts_raw:
                created = dt.date.fromisoformat(post["created_at"][:10])
                if START_DATE <= created <= END_DATE:
                    harvested.append({
                        "id":         post["id"],
                        "url":        f"{BASE_URL}/t/-/{post['topic_id']}/{post['post_number']}",
                        "user":       post["username"],
                        "created_at": post["created_at"],
                        "raw":        post.get("raw", ""),
                        "cooked":     post.get("cooked", ""),
                    })

            time.sleep(RATE_DELAY)     # be kind to the server

    print(f"\n📥  Fetched {len(harvested):,} posts in range {START_DATE} – {END_DATE}")
    return harvested



# ────────────────────────── Main entrypoint ───────────────────────────────────
async def main():
    state = await ensure_storage_state()
    cookie_header = build_cookie_header(state)

    print("🔑  Cookies loaded, starting JSON feed crawl …")
    posts = crawl_posts(cookie_header)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(posts, indent=2), encoding="utf-8")
    print(f"💾  Saved ➜ {OUT_FILE.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())
