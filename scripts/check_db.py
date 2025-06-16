import sqlite3

DB_PATH = "knowledge_base.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("\n TABLE ROW COUNTS:")

for table in ["discourse_chunks", "markdown_chunks"]:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    total = cursor.fetchone()[0]

    cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL")
    with_embeddings = cursor.fetchone()[0]

    print(f"- {table}: {total} rows total, {with_embeddings} rows with embeddings")

conn.close()