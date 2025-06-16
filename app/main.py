"""
app/main.py
──────────────────────────────────────────────────────────────────────────────
FastAPI wrapper around the local RAG.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from .rag import init_rag, answer_question

load_dotenv()               # still fine if you want to keep .env

app = FastAPI(title="TDS Virtual TA (local)")

# initialise once at startup
@app.on_event("startup")
def _startup():
    global RAG_STATE
    try:
        RAG_STATE = init_rag()
        print("✅  RAG index ready.")
    except Exception as e:
        print("❌  Failed to initialise RAG:", e)
        raise


# ─── Schemas ───────────────────────────────────────────────────────────
class Question(BaseModel):
    question: str
    image: str | None = None     # base64 if you add vision later


# ─── Routes ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/")
async def ask(q: Question):
    try:
        return answer_question(RAG_STATE, q.question, q.image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
