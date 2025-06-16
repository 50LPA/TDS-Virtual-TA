#!/usr/bin/env python
"""
Generate local embeddings for markdown_chunks + discourse_chunks
and write them into a FAISS index (faiss.index) — no Internet needed.
"""

from __future__ import annotations
import sqlite3, pathlib, json, numpy as np
from fastembed import TextEmbedding
import faiss, tqdm, gc

DB     = pathlib.Path("knowledge_base.db")
INDEX  = pathlib.Path("faiss.index")
MODEL  = "BAAI/bge-small-en-v1.5"        # Tiny, good quality
DB_BATCH  = 100                          # DB read batch (optional)
EMBED_BATCH = 8                          # Lower this if OOM
EMBED_SUB_BATCH = 2                     # Actual embedder batch size

def rows(conn):
    cur = conn.execute("""
        SELECT id, text FROM markdown_chunks
        UNION ALL
        SELECT id, text FROM discourse_chunks
        WHERE text IS NOT NULL
    """)
    for r in cur:
        yield r

def batch(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def main():
    conn = sqlite3.connect(DB)
    ids, texts = zip(*[(rid, txt) for rid, txt in rows(conn)])

    embedder = TextEmbedding(model_name=MODEL)
    vectors = []
    for i, text_batch in enumerate(tqdm.tqdm(batch(texts, EMBED_BATCH), total=len(texts)//EMBED_BATCH+1, desc="Embedding")):
        try:
            batch_vecs = list(embedder.embed(text_batch, batch_size=EMBED_SUB_BATCH))
            vectors.extend(batch_vecs)
        except Exception as e:
            print(f"[!] Failed at batch {i}: {e}")
            continue
        gc.collect()

    vecs = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    # Save mapping id↔FAISS‑position
    mapping = {i: ids[i] for i in range(len(ids))}
    (INDEX.parent / "faiss_ids.json").write_text(json.dumps(mapping, indent=2))

    faiss.write_index(index, str(INDEX))
    print("✅  FAISS index saved →", INDEX)

if __name__ == "__main__":
    main()
