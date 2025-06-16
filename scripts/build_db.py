#!/usr/bin/env python
"""
scripts/build_db.py
───────────────────────────────────────────────────────────────────────────────
Populate knowledge_base.db with chunked text from:

    • data/course.json      (Markdown pages scraped from GitHub)
    • data/discourse.json   (IIT‑M Online Degree forum posts)

After running you’ll have two tables:

    markdown_chunks   – chunks of course notes
    discourse_chunks  – chunks of Discourse posts

Schema (for both tables)
────────────────────────
id           TEXT  PRIMARY KEY   e.g. "linear-algebra_0", "104123_2"
source_url   TEXT                original URL
chunk_index  INTEGER             0, 1, 2, …
text         TEXT                chunk contents
embedding    BLOB                NULL for now (fill after OpenAI embeddings)

Usage
─────
    python scripts/build_db.py
"""

from __future__ import annotations
import json, sqlite3, textwrap, pathlib, sys

DB_PATH    = "knowledge_base.db"
CHUNK_SIZE = 1000   # characters per chunk


# ──────────────────────────────────────────────────────────────────────────────
def insert_chunks(
    table: str,
    items: list[dict],
    conn: sqlite3.Connection,
    *,
    text_key: str,
) -> None:
    """Create (if needed) `table` and insert chunked rows from `items`."""
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {table} (
                id           TEXT PRIMARY KEY,
                source_url   TEXT,
                chunk_index  INTEGER,
                text         TEXT,
                embedding    BLOB
        )"""
    )

    rows: list[tuple] = []
    for item in items:
        body = item.get(text_key, "") or ""
        if not body.strip():
            continue

        chunks = textwrap.wrap(body, CHUNK_SIZE)
        for idx, chunk in enumerate(chunks):
            rows.append(
                (
                    f"{item['id']}_{idx}",     # unique row ID
                    item.get("url", ""),
                    idx,
                    chunk,
                    None,                     # embedding to be filled later
                )
            )

    if rows:
        conn.executemany(f"INSERT OR REPLACE INTO {table} VALUES (?,?,?,?,?)", rows)
        print(f"  • Inserted {len(rows):,} rows into {table}")
        conn.commit()
    else:
        print(f"  • No rows to insert for {table}")


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    root      = pathlib.Path(__file__).resolve().parents[1]
    db_file   = root / DB_PATH
    data_dir  = root / "data"

    print("📚  Building database …")
    print("→  DB file:", db_file)
    conn = sqlite3.connect(db_file.as_posix())

    # ── Course Markdown pages ────────────────────────────────────────────────
    course_path = data_dir / "course.json"
    if course_path.exists():
        course_items = json.loads(course_path.read_text(encoding="utf-8"))
        print("📝  Loading", course_path)
        insert_chunks("markdown_chunks", course_items, conn, text_key="text")
    else:
        print("⚠️  data/course.json not found – skipping")

    # ── Discourse forum posts ────────────────────────────────────────────────
    discourse_path = data_dir / "discourse.json"
    if discourse_path.exists():
        discourse_items = json.loads(discourse_path.read_text(encoding="utf-8"))
        print("💬  Loading", discourse_path)
        insert_chunks("discourse_chunks", discourse_items, conn, text_key="raw")
    else:
        print("⚠️  data/discourse.json not found – skipping")

    conn.close()
    print("✅  Done – knowledge_base.db is ready.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
