"""
ingest.py — Build the Chroma vector index and BM25 sparse index
from 3GPP specification files in data/specs/.

Supports .docx and .pdf files.  Legacy .doc (OLE2 binary) is listed for
completeness but is not readable by python-docx — convert to .docx or .pdf
before ingesting.

Usage
-----
    python ingest.py                # full ingest (drops and rebuilds)
    python ingest.py --check        # preflight: verify Ollama + Chroma reachable
    python ingest.py --reset        # wipe chroma_db/ and index/ then exit
    python ingest.py --incremental  # skip files whose content hash is unchanged

Indexing pipeline
-----------------
  Parsing:    DocxParser / PDFParser → structured chunk dicts carrying breadcrumb,
              content_type (procedure|table|asn1), section, and release metadata.
  IDs:        SHA-256(spec_id + section_heading + sub_chunk_index), 16 hex chars.
              Stable when chunk_size and parser version are fixed; a config change
              requires a full re-ingest.
  Vectors:    nomic-embed-text via Ollama, stored in a Chroma persistent collection
              with cosine distance.
  Sparse:     BM25-Okapi over alphanumeric tokens, serialised as pickle files in
              index/.  Always rebuilt from the full chunk set (global index).
  Config:     index/ingest_config.json written after every successful run.
              Records embed_model, chunk_size, chunk_overlap, per-file SHA-256
              hashes, and timestamp.  query.py reads this at startup to warn of
              embed-model drift.

Incremental mode
----------------
  Compares each spec file's SHA-256 against the hash stored in ingest_config.json.
  Unchanged files are skipped; their existing Chroma vectors are reused for BM25
  rebuild.  Changed or new files are fully re-parsed and re-embedded; stale chunks
  for that spec_id are deleted from Chroma before the new set is upserted.
  Specs removed from data/specs/ are detected and purged from Chroma automatically.
"""

import re
import sys
import json
import pickle
import shutil
import hashlib
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from tqdm import tqdm
import chromadb
import ollama
from rank_bm25 import BM25Okapi

from parsers.docx_parser import DocxParser
from parsers.pdf_parser import PDFParser

# ── Paths ──────────────────────────────────────────────────────────────────────
SPECS_DIR   = Path("data/specs")
CHROMA_DIR  = Path("chroma_db")
INDEX_DIR   = Path("index")
CONFIG_FILE = INDEX_DIR / "ingest_config.json"

# ── Index constants ────────────────────────────────────────────────────────────
COLLECTION_NAME = "3gpp_specs"
EMBED_MODEL     = "nomic-embed-text"
CHUNK_SIZE      = 1500
CHUNK_OVERLAP   = 300
BATCH_SIZE      = 256   # Chroma write batch size (avoid payload limits)


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_spec_id(filename: str) -> Tuple[str, str]:
    """
    Extract the TS spec ID and human-readable title from a 3GPP filename.

    3GPP file naming convention:
        {spec_number}-{version_code} {title}.{ext}
    Examples:
        "38331-j20 NR Radio Resource Control (RRC); Protocol specification.docx"
            → ("TS 38.331", "NR Radio Resource Control (RRC); Protocol specification")
        "38211-j30_NR Physical channels and modulation.docx"
            → ("TS 38.211", "NR Physical channels and modulation")

    The spec_id format "TS XX.XXX" is the canonical 3GPP identifier used in
    citations (e.g. "as defined in TS 38.331").

    Returns
    -------
    (spec_id, spec_title)
    """
    stem = Path(filename).stem
    m = re.match(r"^(\d{5})-\w+[_ ](.+)$", stem)
    if m:
        num   = m.group(1)                   # e.g. "38331"
        title = m.group(2).replace("_", " ") # normalise underscores
        spec_id = f"TS {num[:2]}.{num[2:]}"  # "TS 38.331"
        return spec_id, title
    # Fallback: use stem as both id and title
    return stem, stem


def tokenize(text: str) -> List[str]:
    """
    Alphanumeric tokenizer for BM25 index construction and query expansion.

    Splits on any non-alphanumeric character and lowercases.  Simple but
    effective for 3GPP text which is dense with camelCase identifiers and
    hyphenated acronyms like 'ssb-SubcarrierOffset'.
    """
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def embed(text: str) -> List[float]:
    """
    Embed a single text string using Ollama nomic-embed-text.

    Tries both 'prompt=' and 'input=' keyword argument variants to handle
    different versions of the ollama Python client.

    Parameters
    ----------
    text : str
        Text to embed (should be ≤ nomic-embed-text's context window, ~8192 tokens).

    Returns
    -------
    List of floats representing the embedding vector.
    """
    try:
        return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
    except TypeError:
        return ollama.embeddings(model=EMBED_MODEL, input=text)["embedding"]


def file_sha256(path: Path) -> str:
    """
    Compute the SHA-256 hash of a file's raw bytes.

    Used for incremental ingest: compare against stored hashes in
    ingest_config.json to determine whether a spec file has changed.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_config() -> Dict:
    """
    Load the ingest configuration written by a previous run.

    Returns an empty dict if no config exists (first-time ingest).
    """
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_config(spec_files: List[Path], total_chunks: int) -> None:
    """
    Persist ingest metadata to index/ingest_config.json.

    Stores chunk_size, chunk_overlap, embed_model, per-file SHA-256 hashes,
    total chunk count, and ISO-8601 timestamp.  query.py reads this to detect
    config drift between the index and the current run settings.

    Parameters
    ----------
    spec_files : list of Path
        Spec files successfully ingested in this run.
    total_chunks : int
        Total number of chunks written to the index.
    """
    config = {
        "chunk_size":    CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_model":   EMBED_MODEL,
        "total_chunks":  total_chunks,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "files": {
            str(p.name): {
                "sha256": file_sha256(p),
                "size_bytes": p.stat().st_size,
            }
            for p in spec_files
        },
    }
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Config saved to {CONFIG_FILE}")


# ── Preflight check (Change 4) ─────────────────────────────────────────────────

def cmd_check() -> None:
    """
    Verify that all external dependencies are reachable before starting ingest.

    Checks:
      1. Ollama is running and nomic-embed-text responds to a test embedding.
      2. Chroma PersistentClient can be instantiated at CHROMA_DIR.
      3. data/specs/ directory exists and contains at least one spec file.

    Exits with code 1 on any failure so CI / shell scripts can detect problems.
    """
    print("── Preflight check ──────────────────────────────────")

    # 1. Ollama embedding model
    print(f"  Ollama ({EMBED_MODEL})... ", end="", flush=True)
    try:
        vec = embed("3GPP preflight test")
        print(f"OK  (vector dim={len(vec)})")
    except Exception as e:
        print(f"FAIL\n  ERROR: {e}")
        print("  Make sure Ollama is running: ollama serve")
        print(f"  And the model is pulled:    ollama pull {EMBED_MODEL}")
        sys.exit(1)

    # 2. Chroma
    print(f"  Chroma ({CHROMA_DIR})...   ", end="", flush=True)
    try:
        CHROMA_DIR.mkdir(exist_ok=True)
        chromadb.PersistentClient(path=str(CHROMA_DIR))
        print("OK")
    except Exception as e:
        print(f"FAIL\n  ERROR: {e}")
        sys.exit(1)

    # 3. Spec files
    print(f"  Specs ({SPECS_DIR})...     ", end="", flush=True)
    if not SPECS_DIR.exists():
        print(f"FAIL\n  Directory not found: {SPECS_DIR}")
        sys.exit(1)
    spec_files = [
        p for p in SPECS_DIR.iterdir()
        if p.suffix.lower() in (".docx", ".doc", ".pdf")
    ]
    if not spec_files:
        print("FAIL\n  No .docx / .pdf files found.")
        sys.exit(1)
    print(f"OK  ({len(spec_files)} file(s) found)")

    print("── All checks passed. Ready to ingest. ─────────────")


# ── Reset (Change 5) ───────────────────────────────────────────────────────────

def cmd_reset() -> None:
    """
    Wipe the Chroma vector store and BM25 index files.

    Ensures a clean state before re-ingesting — prevents stale BM25 pickles
    from a failed or partial previous run staying out of sync with Chroma.

    Prints a summary of what was deleted and exits cleanly.
    """
    print("── Reset ────────────────────────────────────────────")
    deleted = []

    if CHROMA_DIR.exists():
        try:
            shutil.rmtree(CHROMA_DIR)
            deleted.append(str(CHROMA_DIR))
        except PermissionError as exc:
            print(
                f"  WARNING: Could not fully delete {CHROMA_DIR}: {exc}\n"
                "  Close any other process using the Chroma DB and retry."
            )

    for pkl in (INDEX_DIR / "bm25_tokens.pkl", INDEX_DIR / "bm25_ids.pkl", CONFIG_FILE):
        if pkl.exists():
            pkl.unlink()
            deleted.append(str(pkl))

    if deleted:
        for d in deleted:
            print(f"  Deleted: {d}")
        print("── Reset complete. Run python ingest.py to rebuild. ─")
    else:
        print("  Nothing to reset (no existing index found).")


# ── Main ingest ────────────────────────────────────────────────────────────────

def ingest(incremental: bool = False) -> None:
    """
    Build the Chroma vector index and BM25 sparse index from spec files.

    Full ingest (default):
        Drops and recreates the Chroma collection, re-embeds all chunks,
        writes fresh BM25 pickles and ingest_config.json.

    Incremental ingest (--incremental, Change 12):
        Loads previous ingest_config.json to get stored file hashes.
        For each spec file:
          - If hash matches stored hash: skip embedding, reuse Chroma records.
          - If hash differs or file is new: embed all chunks and upsert.
        BM25 is always rebuilt from all chunks (it is a global sparse index).

    Parameters
    ----------
    incremental : bool
        If True, skip re-embedding unchanged spec files.
    """
    CHROMA_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)

    if not SPECS_DIR.exists():
        print(f"ERROR: Specs directory not found: {SPECS_DIR}")
        print("Create data/specs/ and add your 3GPP .docx or .pdf files.")
        sys.exit(1)

    # Note: .doc (legacy Word 97-2003 binary) is listed but python-docx cannot
    # read OLE2 binary format — those files will fail at parse time with an
    # error. Convert to .docx or .pdf before ingesting. See also the
    # OLE/ASN.1 limitation documented in parsers/docx_parser.py.
    spec_files = sorted(
        p for p in SPECS_DIR.iterdir()
        if p.suffix.lower() in (".docx", ".doc", ".pdf")
    )
    if not spec_files:
        print(f"No .docx / .pdf files found in {SPECS_DIR}")
        sys.exit(1)

    print(f"Found {len(spec_files)} spec file(s).\n")

    # ── Load previous config for incremental mode ──────────────────────────
    prev_config = load_config() if incremental else {}
    prev_hashes: Dict[str, str] = {
        name: info["sha256"]
        for name, info in prev_config.get("files", {}).items()
    }

    # ── Parsers ────────────────────────────────────────────────────────────
    docx_parser = DocxParser(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    pdf_parser  = PDFParser(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # ── Chroma ─────────────────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if not incremental:
        # Full ingest: drop and recreate for a clean slate
        try:
            client.delete_collection(COLLECTION_NAME)
            print("[chroma] Deleted existing collection for fresh ingest.")
        except Exception:
            pass
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    else:
        # Incremental: get-or-create (collection may already exist)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[chroma] Incremental mode: reusing existing collection.")

        # ── Removed-spec reconciliation ────────────────────────────────────
        # If a spec file has been deleted from data/specs/ since the last
        # ingest, its chunks may still live in Chroma. Compare spec_ids on
        # disk against every spec_id in the collection and purge stragglers.
        disk_spec_ids = {
            parse_spec_id(p.name)[0]
            for p in spec_files
        }
        chroma_spec_ids: set = set()
        offset = 0
        page_size = 1000
        while True:
            page = collection.get(
                limit=page_size,
                offset=offset,
                include=["metadatas"],
            )
            if not page["ids"]:
                break
            chroma_spec_ids.update(
                m.get("spec_id", "") for m in page["metadatas"]
            )
            if len(page["ids"]) < page_size:
                break
            offset += page_size

        orphaned = chroma_spec_ids - disk_spec_ids - {""}  # remove empty-string sentinel
        if orphaned:
            for orphan_id in sorted(orphaned):
                print(f"[chroma] Purging orphaned spec no longer on disk: {orphan_id}")
                collection.delete(where={"spec_id": orphan_id})
        else:
            print("[chroma] No orphaned specs detected.")

    # ── Accumulate all chunks (for BM25 global rebuild) ────────────────────
    all_ids:     List[str]       = []
    all_docs:    List[str]       = []
    all_metas:   List[Dict]      = []
    bm25_tokens: List[List[str]] = []
    bm25_ids:    List[str]       = []

    # Track which files were actually processed
    processed_files: List[Path] = []

    for spec_path in spec_files:
        spec_id, spec_title = parse_spec_id(spec_path.name)
        release = DocxParser.parse_release_from_filename(spec_path.name)
        current_hash = file_sha256(spec_path)

        # ── Incremental skip check ─────────────────────────────────────────
        if incremental and prev_hashes.get(spec_path.name) == current_hash:
            print(f"[{spec_id}] Unchanged — skipping re-embed (hash match)")
            # Fetch existing chunks for BM25 rebuild.
            # Embeddings are not needed here and are excluded to avoid
            # loading large float arrays that are never used (issue 5 fix).
            existing = collection.get(
                where={"spec_id": spec_id},
                include=["documents", "metadatas"],
            )
            for uid, doc, meta in zip(
                existing["ids"],
                existing["documents"],
                existing["metadatas"],
            ):
                all_ids.append(uid)
                all_docs.append(doc)
                all_metas.append(meta)
                bm25_tokens.append(tokenize(doc))
                bm25_ids.append(uid)
            processed_files.append(spec_path)
            continue

        print(f"[{spec_id}] Release {release} — {spec_title}")

        # ── Parse to structured chunks ──────────────────────────────────────
        raw_chunks: List[Dict] = []
        try:
            if spec_path.suffix.lower() in (".docx", ".doc"):
                raw_chunks = docx_parser.process_document(str(spec_path), spec_id=spec_id)
                stats = docx_parser.heading_stats()
                print(
                    f"  Heading detection — style: {stats['style']}, "
                    f"regex: {stats['regex']}, "
                    f"content paragraphs: {stats['content_paragraphs']}"
                )
            elif spec_path.suffix.lower() == ".pdf":
                pdf_results = pdf_parser.parse_pdf(str(spec_path))
                # Wrap PDF chunks in compatible dict structure
                for r in pdf_results:
                    raw_chunks.append({
                        "text":         r["text"],
                        "chunk_id":     hashlib.sha256(
                            f"{spec_id}\0pdf\0{r['chunk_id']}".encode()
                        ).hexdigest()[:16],
                        "spec_id":      spec_id,
                        "section":      "PDF",
                        "breadcrumb":   spec_id,
                        "content_type": "asn1" if "::=" in r["text"] else "procedure",
                        "chunk_index":  r["chunk_id"],
                        "chunk_length": len(r["text"]),
                    })
        except Exception as e:
            print(f"  WARNING: Failed to parse {spec_path.name}: {e}")
            continue

        if not raw_chunks:
            print(f"  WARNING: No chunks produced from {spec_path.name}")
            continue

        print(f"  {len(raw_chunks)} chunks — embedding...", flush=True)

        # ── Embed and accumulate ────────────────────────────────────────────
        new_ids:        List[str]         = []
        new_docs:       List[str]         = []
        new_metas:      List[Dict]        = []
        new_embeddings: List[List[float]] = []

        for chunk in tqdm(raw_chunks, desc=f"  {spec_id}", leave=True):
            uid       = chunk["chunk_id"]
            chunk_txt = chunk["text"]

            new_ids.append(uid)
            new_docs.append(chunk_txt)
            new_metas.append({
                "spec_id":      spec_id,
                "spec_title":   spec_title,
                "release":      release,
                "source":       spec_path.name,
                "section":      chunk.get("section", ""),
                "breadcrumb":   chunk.get("breadcrumb", ""),
                "content_type": chunk.get("content_type", "procedure"),
                "chunk":        chunk.get("chunk_index", 0),
            })
            new_embeddings.append(embed(chunk_txt))
            bm25_tokens.append(tokenize(chunk_txt))
            bm25_ids.append(uid)

        all_ids.extend(new_ids)
        all_docs.extend(new_docs)
        all_metas.extend(new_metas)
        processed_files.append(spec_path)

        # ── Write this spec's chunks to Chroma ─────────────────────────────
        if incremental:
            # Purge ALL existing chunks for this spec before upserting the
            # new set. Without this, chunks whose IDs changed (different section
            # count, different split boundaries, or parser upgrades) remain in
            # the collection as orphans and can still be retrieved (issue 1 fix).
            try:
                collection.delete(where={"spec_id": spec_id})
                print(f"  Deleted stale chunks for {spec_id} before re-upsert")
            except Exception as e:
                print(f"  WARNING: Could not purge stale chunks for {spec_id}: {e}")

        print(f"  Writing {len(new_ids)} chunks to Chroma...", flush=True)
        for i in range(0, len(new_ids), BATCH_SIZE):
            j = i + BATCH_SIZE
            if incremental:
                collection.upsert(
                    ids=new_ids[i:j],
                    embeddings=new_embeddings[i:j],
                    documents=new_docs[i:j],
                    metadatas=new_metas[i:j],
                )
            else:
                collection.add(
                    ids=new_ids[i:j],
                    embeddings=new_embeddings[i:j],
                    documents=new_docs[i:j],
                    metadatas=new_metas[i:j],
                )

    if not all_ids:
        print("ERROR: No chunks were produced. Check that spec files are readable.")
        sys.exit(1)

    print(f"\n  ✓ Chroma index written to {CHROMA_DIR}/  ({len(all_ids)} total chunks)")

    # ── BM25 — always rebuilt from all chunks (global sparse index) ────────
    print("[bm25] Rebuilding index from all chunks...")
    with open(INDEX_DIR / "bm25_tokens.pkl", "wb") as f:
        pickle.dump(bm25_tokens, f)
    with open(INDEX_DIR / "bm25_ids.pkl", "wb") as f:
        pickle.dump(bm25_ids, f)
    print(f"  ✓ BM25 index written to {INDEX_DIR}/")

    # ── Persist config (Change 6) ──────────────────────────────────────────
    save_config(processed_files, len(all_ids))

    print(
        f"\n✅ Ingestion complete — {len(processed_files)} spec(s), "
        f"{len(all_ids)} total chunks."
    )


# ── CLI entry point ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build and return the argparse CLI parser."""
    p = argparse.ArgumentParser(
        description="3GPP RAG — parse spec files and build Chroma + BM25 indexes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Arguments
---------
  (none)          Full ingest: parse all .docx/.pdf files in data/specs/, embed
                  every chunk with nomic-embed-text, write a fresh Chroma collection
                  (old one is dropped first) and BM25 pickle files.  Writes
                  index/ingest_config.json recording the embed model, chunk
                  settings, file hashes, and timestamp.

  --check         Preflight check: verify that Ollama is running and
                  nomic-embed-text is available, that ChromaDB can be opened, and
                  that at least one spec file exists in data/specs/.  Exits 0 on
                  success, 1 on any failure.  Does NOT modify the index.
                  Run this before a long ingest to catch problems early.

  --reset         Wipe the index: delete chroma_db/ and the BM25 pickle files in
                  index/.  Use this before a full re-ingest when you have changed
                  CHUNK_SIZE, CHUNK_OVERLAP, or the embedding model — changes to
                  those parameters cause chunk ID drift and require a clean rebuild.

  --incremental   Skip re-embedding spec files whose SHA-256 hash matches the hash
                  stored from the last run.  Only changed or new files are
                  re-parsed and re-embedded; stale chunks for those specs are
                  deleted from Chroma before the new set is upserted.  Specs
                  removed from data/specs/ are also purged automatically.
                  BM25 is always rebuilt from all chunks (global index).
                  NOTE: incremental mode does NOT detect chunk_size/model changes;
                  if you changed indexing parameters, run --reset first.

Typical workflows
-----------------
  First-time setup:
    python ingest.py --check && python ingest.py

  Add or update a spec file without re-embedding everything:
    python ingest.py --incremental

  Rebuild after changing CHUNK_SIZE or EMBED_MODEL:
    python ingest.py --reset && python ingest.py
""",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Verify Ollama + Chroma are reachable and specs exist, then exit (does not modify the index).",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Delete chroma_db/ and BM25 index files, then exit.  Required when chunk_size or embed model changes.",
    )
    p.add_argument(
        "--incremental",
        action="store_true",
        help="Re-embed only files whose content has changed since the last ingest (detected by SHA-256 hash).",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.check:
        cmd_check()
    elif args.reset:
        cmd_reset()
    else:
        ingest(incremental=args.incremental)
