# TDS Virtual TA

A local **Retrieval-Augmented Generation (RAG)** application that answers student questions using course content and Discourse posts from the **Tools in Data Science (TDS)** course offered by IIT Madras.

This project uses:
- **FastAPI** for API backend
- **FAISS** for vector search
- **fastembed** for embeddings
- **AIPipe** as the LLM provider
- **Promptfoo** for evaluation (optional)

---

## Features

- Answers questions from TDS students using course materials and discussion forum.
- Uses both course notes and Discourse posts for retrieval.
- Optional vision support via base64 images.
- Supports local evaluation with [Promptfoo](https://promptfoo.dev/).

---

## Project Structure

```txt
project-root/
├── app/
│   ├── main.py         # FastAPI server
│   └── rag.py          # RAG logic (embed, search, query LLM)
├── embed_local.py      # Script to generate FAISS index and DB
├── .env                # Environment variables
├── faiss.index         # Vector index (generated)
├── faiss_ids.json      # Mapping between FAISS IDs and chunk IDs
├── knowledge_base.db   # SQLite DB with chunks
└── README.md           # This file