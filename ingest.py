"""
ingest.py — Build the Chroma vector index and BM25 sparse index
from 3GPP specification files in data/specs/.

Supports .docx and .pdf files.
Run once (or any time specs are added/updated):
    python ingest.py
"""

import re
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

from tqdm import tqdm
import chromadb
import ollama
from rank_bm25 import BM25Okapi

from parsers.docx_parser import DocxParser
from parsers.pdf_parser import PDFParser

# ── Paths ──────────────────────────────────────────────────────────────────────
SPECS_DIR       = Path("data/specs")
CHROMA_DIR      = Path("chroma_db")
INDEX_DIR       = Path("index")

CHROMA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

COLLECTION_NAME = "3gpp_specs"
EMBED_MODEL     = "nomic-embed-text"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
BATCH_SIZE      = 256


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_spec_id(filename: str) -> Tuple[str, str]:
    """
    Extract the TS spec ID and title from a 3GPP filename.

    Examples:
        '38331-j20 NR Radio Resource Control (RRC); Protocol specification.docx'
        → ('TS 38.331', 'NR Radio Resource Control (RRC); Protocol specification')

        '38211-j30_NR Physical channels and modulation.docx'
        → ('TS 38.211', 'NR Physical channels and modulation')
    """
    stem = Path(filename).stem
    m = re.match(r"^(\d{5})-\w+[_ ](.+)$", stem)
    if m:
        num   = m.group(1)                      # e.g. '38331'
        title = m.group(2).replace("_", " ")    # clean underscores
        spec_id = f"TS {num[:2]}.{num[2:]}"     # 'TS 38.331'
        return spec_id, title
    return stem, stem


def tokenize(text: str) -> List[str]:
    """Alphanumeric tokenizer for BM25."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def embed(text: str) -> List[float]:
    """Embed a single string via Ollama nomic-embed-text."""
    try:
        return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
    except TypeError:
        return ollama.embeddings(model=EMBED_MODEL, input=text)["embedding"]


# ── Main ingest ────────────────────────────────────────────────────────────────

def ingest() -> None:
    if not SPECS_DIR.exists():
        print(f"ERROR: Specs directory not found: {SPECS_DIR}")
        print("Create data/specs/ and add your 3GPP .docx or .pdf files.")
        sys.exit(1)

    spec_files = sorted(
        p for p in SPECS_DIR.iterdir()
        if p.suffix.lower() in (".docx", ".doc", ".pdf")
    )
    if not spec_files:
        print(f"No .docx / .pdf files found in {SPECS_DIR}")
        sys.exit(1)

    print(f"Found {len(spec_files)} spec file(s).\n")

    docx_parser = DocxParser(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    pdf_parser  = PDFParser(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Chroma — drop and recreate for clean re-runs
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
        print("[chroma] Deleted existing collection for fresh ingest.")
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_ids:        List[str]         = []
    all_docs:       List[str]         = []
    all_metas:      List[Dict]        = []
    all_embeddings: List[List[float]] = []
    bm25_tokens:    List[List[str]]   = []
    bm25_ids:       List[str]         = []

    for spec_path in spec_files:
        spec_id, spec_title = parse_spec_id(spec_path.name)
        print(f"[{spec_id}] {spec_title}")

        # Parse to raw text chunks
        raw_chunks: List[str] = []
        try:
            if spec_path.suffix.lower() in (".docx", ".doc"):
                processed  = docx_parser.process_document(str(spec_path))
                raw_chunks = [c["text"] for c in processed]
            elif spec_path.suffix.lower() == ".pdf":
                processed  = pdf_parser.parse_pdf(str(spec_path))
                raw_chunks = [c["text"] for c in processed]
        except Exception as e:
            print(f"  WARNING: Failed to parse {spec_path.name}: {e}")
            continue

        print(f"  {len(raw_chunks)} chunks — embedding...", flush=True)

        for idx, chunk_text in enumerate(tqdm(raw_chunks, desc=f"  {spec_id}", leave=True)):
            uid = f"{spec_id.replace(' ', '_').replace('.', '')}__chunk-{idx:05d}"

            all_ids.append(uid)
            all_docs.append(chunk_text)
            all_metas.append({
                "spec_id":    spec_id,
                "spec_title": spec_title,
                "source":     spec_path.name,
                "chunk":      idx,
            })
            all_embeddings.append(embed(chunk_text))
            bm25_tokens.append(tokenize(chunk_text))
            bm25_ids.append(uid)

    if not all_ids:
        print("ERROR: No chunks were produced. Check that spec files are readable.")
        sys.exit(1)

    # Write to Chroma in batches
    print(f"\n[chroma] Storing {len(all_ids)} chunks...")
    for i in range(0, len(all_ids), BATCH_SIZE):
        j = i + BATCH_SIZE
        collection.add(
            ids=all_ids[i:j],
            embeddings=all_embeddings[i:j],
            documents=all_docs[i:j],
            metadatas=all_metas[i:j],
        )
    print(f"  ✓ Chroma index written to {CHROMA_DIR}/")

    # Write BM25 index
    print("[bm25] Saving index...")
    with open(INDEX_DIR / "bm25_tokens.pkl", "wb") as f:
        pickle.dump(bm25_tokens, f)
    with open(INDEX_DIR / "bm25_ids.pkl", "wb") as f:
        pickle.dump(bm25_ids, f)
    print(f"  ✓ BM25 index written to {INDEX_DIR}/")

    print(f"\n✅ Ingestion complete — {len(spec_files)} spec(s), {len(all_ids)} total chunks.")


if __name__ == "__main__":
    ingest()
