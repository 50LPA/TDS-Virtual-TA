from __future__ import annotations
import sys
import json, os, sqlite3, numpy as np, requests, textwrap
from pathlib import Path
from dotenv import load_dotenv
import faiss
from fastembed import TextEmbedding

# ─── Load env ───────────────────────────────────────────────────────────
load_dotenv()                   # reads .env
AIPIPE_KEY   = os.getenv("AIPIPE_API_KEY")
BASE_URL     = os.getenv("AIPIPE_BASE_URL", "https://api.aipipe.ai/v1").rstrip("/")
MODEL_NAME   = os.getenv("CHAT_MODEL", "llama3-8b-instruct")
DEBUG        = bool(int(os.getenv("RAG_DEBUG", "0")))

API_URL      = f"{BASE_URL}/chat/completions"
HEADERS      = {
    "Authorization": f"Bearer {AIPIPE_KEY}",
    "Content-Type":  "application/json",
}

# ─── Retrieval config ──────────────────────────────────────────────────
DB_PATH   = Path("knowledge_base.db")
INDEX_BIN = Path("faiss.index")
ID_MAP    = Path("faiss_ids.json")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K       = 6


# ─── Startup ────────────────────────────────────────────────────────────
def init_rag() -> dict:
    if not (INDEX_BIN.exists() and ID_MAP.exists()):
        raise RuntimeError("FAISS index or ID map missing – run embed_local.py first")

    conn     = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    index    = faiss.read_index(str(INDEX_BIN))
    id_map   = json.loads(ID_MAP.read_text())
    embedder = TextEmbedding(model_name=EMBED_MODEL)

    return {"db": conn, "index": index, "id_map": id_map, "embed": embedder}


# ─── Internal Helpers ──────────────────────────────────────────────────
def _retrieve(state: dict, query: str):
    q_vec = np.array(list(state["embed"].embed([query]))[0], dtype="float32")[None, :]
    _, I  = state["index"].search(q_vec, TOP_K)
    ids   = [state["id_map"][str(i)] for i in I[0]]

    cur = state["db"].execute(
        f"""
        SELECT id, text, source_url
          FROM markdown_chunks WHERE id IN ({','.join('?'*len(ids))})
        UNION ALL
        SELECT id, text, source_url
          FROM discourse_chunks WHERE id IN ({','.join('?'*len(ids))})
        """,
        ids * 2,
    )
    return cur.fetchall()


import sys

def _ask_ai_pipe(prompt: str) -> str:
    if not AIPIPE_KEY:
        raise RuntimeError("AIPIPE_API_KEY is missing")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful TA for IIT‑M’s Tools in Data Science course."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=45)

        if DEBUG:
            sys.stderr.write(f"Status: {resp.status_code}\n")
            sys.stderr.write(f"Raw: {resp.text[:1000]}\n")

        resp.raise_for_status()
        out = resp.json()

        if "choices" in out and "message" in out["choices"][0]:
            return out["choices"][0]["message"]["content"].strip()

        if "result" in out:
            return out["result"].strip()

        raise ValueError(f"Unexpected format: {json.dumps(out)[:400]}")

    except Exception as e:
        sys.stderr.write(f"AI Pipe error: {str(e)}\n")
        raise


# ─── Public API ─────────────────────────────────────────────────────────
def answer_question(state: dict, question: str, image: str | None = None):
    passages = _retrieve(state, question)
    links    = [{"url": p["source_url"], "text": ""} for p in passages]

    if not passages:
        return {"answer": "I couldn't find any relevant documents.", "links": links}

    context = "\n\n".join(p["text"] for p in passages[:TOP_K])

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely in 3‑4 sentences. "
        "Reference the passages when useful."
    )

    try:
        answer = _ask_ai_pipe(prompt)
    except Exception as e:
        if DEBUG:
            print("AI Pipe error:", e)
        answer = (
            "Sorry, I ran into an error while generating the answer. "
            "Here are the raw passages I found:\n\n---\n\n" + context[:1500]
        )

    return {"answer": answer, "links": links}
