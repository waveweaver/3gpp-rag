"""
query.py — Interactive 3GPP Standards Q&A system.

Uses hybrid retrieval (BM25 sparse + Chroma vector) fused with
Reciprocal Rank Fusion (RRF), answered by Ollama llama3.3.

Requires a built index — run ingest.py first.
    python query.py
"""

import re
import sys
import pickle
from pathlib import Path
from typing import List
from collections import defaultdict

import chromadb
import ollama
from rank_bm25 import BM25Okapi

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_DIR      = Path("chroma_db")
INDEX_DIR       = Path("index")
COLLECTION_NAME = "3gpp_specs"
EMBED_MODEL     = "nomic-embed-text"
LLM_MODEL       = "llama3.2:3b"

K_EACH  = 6   # candidates fetched from each retriever (BM25 + vector)
FINAL_K = 5   # chunks sent to LLM after RRF fusion


# ── Helpers ────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def embed(text: str) -> List[float]:
    try:
        return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
    except TypeError:
        return ollama.embeddings(model=EMBED_MODEL, input=text)["embedding"]


def rrf_merge(list_a: List[str], list_b: List[str], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion of two ranked ID lists."""
    scores: dict = defaultdict(float)
    for lst in (list_a, list_b):
        for rank, uid in enumerate(lst):
            scores[uid] += 1.0 / (k + rank + 1)
    return [uid for uid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:FINAL_K]


def load_indexes():
    bm25_pkl = INDEX_DIR / "bm25_tokens.pkl"
    ids_pkl  = INDEX_DIR / "bm25_ids.pkl"
    if not bm25_pkl.exists() or not ids_pkl.exists():
        print("ERROR: Index not found. Run  python ingest.py  first.")
        sys.exit(1)
    with open(bm25_pkl, "rb") as f:
        tokens = pickle.load(f)
    with open(ids_pkl, "rb") as f:
        ids = pickle.load(f)
    return BM25Okapi(tokens), ids


# ── Core Q&A ───────────────────────────────────────────────────────────────────

def answer(query: str, collection, bm25: BM25Okapi, bm25_ids: List[str]) -> None:
    # 1. BM25 sparse retrieval
    q_tokens     = tokenize(query)
    scores       = bm25.get_scores(q_tokens)
    bm25_top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:K_EACH]
    bm25_top_ids = [bm25_ids[i] for i in bm25_top_idx]

    # 2. Chroma vector retrieval
    q_emb    = embed(query)
    vec_res  = collection.query(query_embeddings=[q_emb], n_results=K_EACH)
    vec_ids  = vec_res["ids"][0]

    # 3. RRF fusion
    fused_ids = rrf_merge(bm25_top_ids, vec_ids)

    # 4. Fetch fused chunks
    fetched    = collection.get(ids=fused_ids)
    id_to_doc  = dict(zip(fetched["ids"], fetched["documents"]))
    id_to_meta = dict(zip(fetched["ids"], fetched["metadatas"]))

    # 5. Build context with spec headers
    sections = []
    for uid in fused_ids:
        meta   = id_to_meta[uid]
        header = f"[{meta['spec_id']} — {meta['spec_title']}, chunk {meta['chunk']}]"
        sections.append(f"{header}\n{id_to_doc[uid]}")
    context = "\n\n---\n\n".join(sections)

    # 6. LLM generation
    system_prompt = (
        "You are a precise technical assistant specializing in 3GPP wireless standards. "
        "Answer ONLY using the provided 3GPP specification context. "
        "Always cite the spec ID (e.g. TS 38.331) when referencing procedures or requirements. "
        "If the answer cannot be found in the provided context, say so clearly."
    )
    user_prompt = (
        f"Context from 3GPP specifications:\n\n{context}\n\nQuestion: {query}"
    )

    print("\nAnswer:")
    print("─" * 60)
    try:
        stream = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": 0.1},
            stream=True,
        )
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
    except Exception as e:
        print(f"ERROR calling LLM: {e}")
        return

    # 7. Print source citations
    print("\n\nSources:")
    print("─" * 60)
    seen: set = set()
    for uid in fused_ids:
        meta  = id_to_meta[uid]
        label = f"  • {meta['spec_id']} — {meta['spec_title']}  (chunk {meta['chunk']})"
        if label not in seen:
            print(label)
            seen.add(label)
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading indexes...", end=" ", flush=True)
    bm25, bm25_ids = load_indexes()

    chroma_db_file = CHROMA_DIR / "chroma.sqlite3"
    if not chroma_db_file.exists():
        print("\nERROR: Chroma DB not found. Run  python ingest.py  first.")
        sys.exit(1)

    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    print("ready.\n")

    print("=" * 60)
    print("  3GPP Standards RAG & QA System")
    print(f"  Retrieval : BM25-Okapi + nomic-embed-text → RRF (k={FINAL_K})")
    print(f"  Generation: {LLM_MODEL} via Ollama (local)")
    print("=" * 60)
    print("Type your question, or 'exit' to quit.\n")

    while True:
        try:
            query = input("Query: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        answer(query, collection, bm25, bm25_ids)


if __name__ == "__main__":
    main()
