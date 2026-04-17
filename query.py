"""
query.py — Interactive 3GPP Standards Q&A system.

Uses hybrid retrieval (BM25 sparse + Chroma vector) fused with
Reciprocal Rank Fusion (RRF), answered by Ollama llama3.2:3b.

Requires a built index — run ingest.py first.

Usage
-----
    python query.py                           # interactive mode
    python query.py --stats                   # print index statistics and exit
    python query.py --debug                   # interactive + verbose retrieval trace
    python query.py --filter-type procedure   # restrict to procedure chunks
    python query.py --filter-type table       # restrict to table chunks
    python query.py --filter-type asn1        # restrict to ASN.1 chunks

Spec-scoped queries (inline prefix):
    Query: [TS 38.331]: What is the MasterInformationBlock?
    (restricts both BM25 and vector retrieval to chunks from TS 38.331 only)

Retrieval pipeline
------------------
  1. Query expansion: acronyms and known 3GPP internal identifiers are appended
     to the query before tokenisation and embedding to bridge term mismatch
     between user shorthand and spec prose (e.g. "MIB" → "Master Information
     Block", "handover" → "reconfigurationWithSync").
  2. BM25 sparse retrieval: top-K candidates by Okapi BM25 score.
     Rows with zero score (no shared tokens) are dropped before fusion;
     if all rows are zero the filter is bypassed so the vector side can
     still contribute through RRF.
  3. Vector retrieval: top-K candidates by cosine distance (nomic-embed-text).
  4. RRF fusion: Reciprocal Rank Fusion merges both ranked lists.
  5. LLM generation: fused context passed to llama3.2:3b with a strict
     grounding prompt — the model cites only section paths present in the
     retrieved context and explicitly states when the answer is not found.
"""

import re
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict

import chromadb
import ollama
from rank_bm25 import BM25Okapi

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_DIR      = Path("chroma_db")
INDEX_DIR       = Path("index")
CONFIG_FILE     = INDEX_DIR / "ingest_config.json"
COLLECTION_NAME = "3gpp_specs"
EMBED_MODEL     = "nomic-embed-text"
LLM_MODEL       = "llama3.2:3b"
CHUNK_SIZE      = 1500   # must match value used at ingest time
CHUNK_OVERLAP   = 300    # must match value used at ingest time

K_EACH  = 10   # candidates fetched from each retriever (BM25 + vector)
FINAL_K = 5    # chunks sent to LLM after RRF fusion


# ── Query expansion dictionary ────────────────────────────────────────────────
#
# Two categories of entries:
#
# 1. Acronym ↔ full-form pairs (bidirectional): bridges user shorthand to spec
#    prose and vice-versa (e.g. "MIB" ↔ "Master Information Block").
#    Source: 3GPP TS/TR glossary and common RAN1-RAN3 usage.
#
# 2. Natural-language term → internal 3GPP identifier: bridges the gap between
#    how engineers phrase questions and the verbatim tokens that appear in
#    procedure clause bodies (e.g. "handover" → "reconfigurationWithSync").
#    Only single-token or hyphenated keys are used here — multi-word phrase
#    keys are not supported by the per-token expansion walk.
#    Keep this list small and query-validated; over-bridging narrows recall.
ACRONYM_MAP = {
    # Physical layer
    "BWP":   "Bandwidth Part",
    "SSB":   "Synchronization Signal Block",
    "CSI":   "Channel State Information",
    "PDCCH": "Physical Downlink Control Channel",
    "PDSCH": "Physical Downlink Shared Channel",
    "PUCCH": "Physical Uplink Control Channel",
    "PUSCH": "Physical Uplink Shared Channel",
    "PRACH": "Physical Random Access Channel",
    "SRS":   "Sounding Reference Signal",
    "DM-RS": "Demodulation Reference Signal",
    "DMRS":  "Demodulation Reference Signal",
    "PT-RS": "Phase-Tracking Reference Signal",
    "PTRS":  "Phase-Tracking Reference Signal",
    "NZP":   "Non-Zero Power",
    # MAC
    "BSR":   "Buffer Status Report",
    # SR omitted: collides with common English usage ("sr", "Sr.")
    "PHR":   "Power Headroom Report",
    "HARQ":  "Hybrid Automatic Repeat Request",
    # RRC
    "MIB":   "Master Information Block",
    "SIB":   "System Information Block",
    "SIB1":  "System Information Block Type 1",
    "RRC":   "Radio Resource Control",
    # AS and NAS omitted: "AS" collides with the common English word
    "NAS":   "Non-Access Stratum",
    "MCG":   "Master Cell Group",
    "SCG":   "Secondary Cell Group",
    "PCell": "Primary Cell",
    "SCell": "Secondary Cell",
    "SpCell": "Special Cell",
    "PSCell": "Primary Secondary Cell",
    # Handover / mobility
    "HO":    "Handover",
    "CHO":   "Conditional Handover",
    "DAPS":  "Dual Active Protocol Stack",
    "RACH":  "Random Access Channel",
    # Natural-language → 3GPP internal identifier bridges.
    # These map common query terms to the verbatim tokens that appear in
    # procedure clause bodies, improving BM25 term matching for conceptual
    # questions where the user uses plain English and the spec uses ASN.1-style
    # identifiers.  Validated against TS 38.331 clause text.
    "handover":         "reconfigurationWithSync",
    "re-establishment": "RRCReestablishment",
    "reestablishment":  "RRCReestablishment",
    "reconfiguration":  "RRCReconfiguration",
    "measurement-gap":  "measGapConfig",
    "beam-failure":     "beamFailureRecovery",
    # Network architecture
    "UE":    "User Equipment",
    "gNB":   "Next-Generation Node B",
    "AMF":   "Access and Mobility Management Function",
    "SMF":   "Session Management Function",
    "UPF":   "User Plane Function",
    "NG-RAN": "Next Generation Radio Access Network",
    "E-UTRA": "Evolved Universal Terrestrial Radio Access",
    # NR omitted: too short / collides with "nr" as English abbreviation
    "LTE":   "Long Term Evolution",
    "EPC":   "Evolved Packet Core",
    "5GC":   "5G Core Network",
    # Measurements
    "RSRP":  "Reference Signal Received Power",
    "RSRQ":  "Reference Signal Received Quality",
    "SINR":  "Signal to Interference plus Noise Ratio",
    "SSTD":  "System Synchronisation Time Difference",
    # Positioning
    "PRS":   "Positioning Reference Signal",
    "DL-TDOA": "Downlink Time Difference of Arrival",
    "UL-AoA":  "Uplink Angle of Arrival",
    # General
    "IE":    "Information Element",
    "ASN1":  "Abstract Syntax Notation One",
    "ASN.1": "Abstract Syntax Notation One",
    "QoS":   "Quality of Service",
    "DRB":   "Data Radio Bearer",
    "SRB":   "Signalling Radio Bearer",
}

# Inverted map for full-form → acronym expansion
_REVERSE_MAP = {v.lower(): k for k, v in ACRONYM_MAP.items()}

# Multi-word phrase → expansion mapping.
# Keys are lowercase substrings matched against the whole query before the
# per-token walk; values are whitespace-separated tokens appended verbatim.
# Listed longest-first so more specific phrases shadow shorter overlaps.
PHRASE_BRIDGES: dict = {
    "handover failure": "T304 handoverFailure",
    "ho failure":       "T304 handoverFailure",
    "link failure":     "radioLinkFailure rlf",
    "random access":    "RACH rachProcedure",
    "cell reselection": "cellReselection s-Measure",
}


def expand_acronyms(query: str) -> str:
    """
    Expand 3GPP acronyms and known phrase patterns in the query string.

    Processing order:
    1. Phrase pre-scan (PHRASE_BRIDGES): multi-word substrings matched against
       the lowercased query before tokenisation so that compound terms are
       bridged to their internal identifiers.
    2. Acronym → full form (ACRONYM_MAP): each token looked up by exact key,
       then uppercased key, so both 'mib' and 'MIB' expand to the full form.
    3. Full form → acronym (_REVERSE_MAP): scanned once against the entire
       query string to catch bidirectional expansion.

    Returns the expanded query string, or the original if nothing matched.
    """
    extras: list = []
    seen:   set  = set()
    query_lower = query.lower()
    query_upper = query.upper()

    # 1. Phrase bridges (multi-word keys — handled before per-token walk)
    for phrase, expansion in PHRASE_BRIDGES.items():
        if phrase in query_lower:
            for token in expansion.split():
                if token not in seen:
                    extras.append(token)
                    seen.add(token)

    # 2. Acronym → full form: try exact key, then uppercased key
    words = re.findall(r"[A-Za-z0-9.\-]+", query)
    for w in words:
        full_form = ACRONYM_MAP.get(w) or ACRONYM_MAP.get(w.upper())
        if full_form and full_form not in seen:
            extras.append(full_form)
            seen.add(full_form)

    # 3. Full form → acronym: scan query once (outside the per-word loop
    # to avoid O(words × map) redundant iteration)
    for full, acr in _REVERSE_MAP.items():
        if full in query_lower and acr not in query_upper and acr not in seen:
            extras.append(acr)
            seen.add(acr)

    if extras:
        return f"{query} {' '.join(extras)}"
    return query


# ── Helpers ────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Alphanumeric tokenizer used for BM25 and acronym-expanded query."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def embed(text: str) -> List[float]:
    """Embed text via Ollama nomic-embed-text."""
    try:
        return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
    except TypeError:
        return ollama.embeddings(model=EMBED_MODEL, input=text)["embedding"]


def rrf_merge(
    list_a: List[str], list_b: List[str], k: int = 60, top_n: int = FINAL_K
) -> List[str]:
    """Reciprocal Rank Fusion of two ranked ID lists."""
    scores: dict = defaultdict(float)
    for lst in (list_a, list_b):
        for rank, uid in enumerate(lst):
            scores[uid] += 1.0 / (k + rank + 1)
    return [uid for uid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:top_n]


def load_indexes():
    """Load BM25 index pickles; exit with helpful message if missing."""
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


def parse_scope_prefix(query: str) -> Tuple[str, Optional[str]]:
    """
    Extract an optional spec-scope prefix from the query.

    Syntax: "[TS 38.331]: actual question"

    Returns (clean_query, spec_id) where spec_id is None when no prefix present.
    """
    m = re.match(r"^\[([^\]]+)\]:\s*(.+)$", query.strip(), re.DOTALL)
    if m:
        spec_id     = m.group(1).strip()   # e.g. "TS 38.331"
        clean_query = m.group(2).strip()
        return clean_query, spec_id
    return query, None


def load_config() -> dict:
    """Load ingest_config.json; return empty dict if absent."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _check_index_config() -> None:
    """
    Compare query.py's EMBED_MODEL, CHUNK_SIZE, and CHUNK_OVERLAP constants
    against the values recorded in ingest_config.json.  Prints a warning for
    each mismatch so users can catch configuration drift between ingest and
    query runs.

    Non-fatal: the system continues, because the user may intentionally be
    testing a new embedding model against an old index.
    """
    config = load_config()
    if not config:
        return  # no config yet (first run before any ingest)
    indexed_model = config.get("embed_model", "")
    if indexed_model and indexed_model != EMBED_MODEL:
        print(
            f"WARNING: embed model mismatch.\n"
            f"  Index was built with : {indexed_model}\n"
            f"  query.py is using    : {EMBED_MODEL}\n"
            f"  Query embeddings will not match index vectors. "
            f"Re-ingest with the same model or update EMBED_MODEL in query.py."
        )
    for param, local_val in [("chunk_size", CHUNK_SIZE), ("chunk_overlap", CHUNK_OVERLAP)]:
        stored_val = config.get(param)
        if stored_val is not None and stored_val != local_val:
            print(
                f"WARNING: {param} mismatch.\n"
                f"  Index was built with : {stored_val}\n"
                f"  query.py constant    : {local_val}\n"
                f"  Chunk boundaries differ; retrieval may be sub-optimal. "
                f"Re-ingest or align the constant in query.py."
            )


# ── Stats command ─────────────────────────────────────────────────────────────

def cmd_stats(collection) -> None:
    """
    Print index statistics and ingest configuration to stdout, then exit.

    Reads ingest_config.json (written by ingest.py) for chunk settings and
    per-file metadata, and queries Chroma for live collection stats.
    """
    config = load_config()
    count  = collection.count()

    print("=" * 60)
    print("  3GPP RAG Index Statistics")
    print("=" * 60)
    print(f"  Total chunks in Chroma : {count}")

    # Distinct specs and releases from Chroma — paginated to handle large indexes
    all_meta: list = []
    _page, _page_size = 0, 1000
    while True:
        page  = collection.get(limit=_page_size, offset=_page, include=["metadatas"])
        batch = page["metadatas"]
        if not batch:
            break
        all_meta.extend(batch)
        if len(batch) < _page_size:
            break
        _page += _page_size
    specs    = sorted({m.get("spec_id", "?")  for m in all_meta})
    releases = sorted({m.get("release",  "?") for m in all_meta})
    ctypes   = sorted({m.get("content_type", "?") for m in all_meta})

    print(f"  Distinct specs         : {len(specs)}")
    for s in specs:
        spec_chunks = sum(1 for m in all_meta if m.get("spec_id") == s)
        print(f"    • {s} ({spec_chunks} chunks)")

    print(f"  Releases               : {', '.join(releases)}")
    print(f"  Content types          : {', '.join(ctypes)}")

    if config:
        print(f"  Ingest timestamp       : {config.get('timestamp', 'n/a')}")
        print(f"  Chunk size / overlap   : {config.get('chunk_size')} / {config.get('chunk_overlap')}")
        print(f"  Embed model            : {config.get('embed_model')}")
        print()
        print("  Source files:")
        for fname, info in config.get("files", {}).items():
            size_kb = info.get("size_bytes", 0) // 1024
            print(f"    • {fname} ({size_kb} KB)")
    else:
        print("  (No ingest_config.json found in index/ — run ingest.py to generate)")

    print("=" * 60)


# ── Core Q&A ───────────────────────────────────────────────────────────────────

def answer(
    query: str,
    collection,
    bm25: BM25Okapi,
    bm25_ids: List[str],
    bm25_metas: List[dict],
    debug: bool = False,
    filter_type: Optional[str] = None,
) -> None:
    """
    Run hybrid retrieval + LLM generation for a single query.

    Parameters
    ----------
    query : str
        Raw user query string (may contain spec-scope prefix and/or acronyms).
    collection : chromadb.Collection
        The Chroma collection to query.
    bm25 : BM25Okapi
        Pre-built BM25 index over all chunk token lists.
    bm25_ids : list of str
        Chunk IDs corresponding to BM25 rows.
    bm25_metas : list of dict
        Metadata dicts for each BM25 row; used for spec/type filtering.
    debug : bool
        If True, print retrieval diagnostics for BM25 and vector stages.
    filter_type : str | None
        One of 'procedure', 'table', 'asn1'.  When set, restricts retrieval
        to chunks with that content_type value.
    """
    # ── Parse spec-scope prefix ─────────────────────────────────────────────
    clean_query, scope_spec_id = parse_scope_prefix(query)

    # ── Acronym expansion ───────────────────────────────────────────────────
    expanded_query = expand_acronyms(clean_query)
    if debug and expanded_query != clean_query:
        print(f"\n[debug] Expanded query : {expanded_query!r}")

    # ── Build Chroma where-clause ───────────────────────────────────────────
    where_clause: Optional[dict] = None
    filters = {}
    if scope_spec_id:
        filters["spec_id"] = scope_spec_id
    if filter_type:
        filters["content_type"] = filter_type

    if len(filters) == 1:
        where_clause = filters
    elif len(filters) == 2:
        where_clause = {"$and": [{k: v} for k, v in filters.items()]}

    # ── 1. BM25 sparse retrieval ────────────────────────────────────────────
    q_tokens = tokenize(expanded_query)
    scores   = bm25.get_scores(q_tokens)

    # Build the candidate pool, applying spec/type filter when active.
    candidate_indices = list(range(len(scores)))
    if scope_spec_id or filter_type:
        candidate_indices = [
            i for i, meta in enumerate(bm25_metas)
            if (not scope_spec_id or meta.get("spec_id") == scope_spec_id)
            and (not filter_type or meta.get("content_type") == filter_type)
        ]

    candidate_indices.sort(key=lambda i: scores[i], reverse=True)

    # Drop rows where BM25 score is zero — those share no tokens with the
    # query and add noise to RRF.  If every candidate scores zero (e.g. a
    # purely numeric query or full tokenizer mismatch), fall back to the
    # full sorted list so the vector side can still contribute through RRF.
    nonzero_indices = [i for i in candidate_indices if scores[i] > 0.0]
    bm25_top_idx = (nonzero_indices if nonzero_indices else candidate_indices)[:K_EACH]

    bm25_top_ids = [bm25_ids[i] for i in bm25_top_idx]

    if debug:
        print("\n[debug] ── BM25 top results ─────────────────────────────")
        for rank, idx in enumerate(bm25_top_idx[:5]):
            uid   = bm25_ids[idx]
            score = scores[idx]
            meta  = bm25_metas[idx]
            # Fetch text preview from Chroma
            try:
                snap = collection.get(ids=[uid], include=["documents"])
                preview = snap["documents"][0][:120].replace("\n", " ")
            except Exception:
                preview = "(unavailable)"
            print(
                f"  [{rank+1}] score={score:.4f}  id={uid}\n"
                f"       spec={meta.get('spec_id')} type={meta.get('content_type')}\n"
                f"       preview: {preview}…"
            )

    # ── 2. Chroma vector retrieval ──────────────────────────────────────────
    q_emb   = embed(expanded_query)
    vec_kwargs: dict = dict(query_embeddings=[q_emb], n_results=K_EACH, include=["distances", "metadatas", "documents"])
    if where_clause:
        vec_kwargs["where"] = where_clause
    vec_res = collection.query(**vec_kwargs)
    vec_ids = vec_res["ids"][0]

    if debug:
        print("\n[debug] ── Vector top results ───────────────────────────")
        for rank, (uid, dist, meta, doc) in enumerate(zip(
            vec_ids,
            vec_res["distances"][0],
            vec_res["metadatas"][0],
            vec_res["documents"][0],
        )):
            preview = doc[:120].replace("\n", " ")
            print(
                f"  [{rank+1}] dist={dist:.4f}  id={uid}\n"
                f"       spec={meta.get('spec_id')} type={meta.get('content_type')}\n"
                f"       breadcrumb: {meta.get('breadcrumb', '')}\n"
                f"       preview: {preview}…"
            )

    # ── 3. RRF fusion ───────────────────────────────────────────────────────
    fused_ids = rrf_merge(bm25_top_ids, vec_ids)

    if debug:
        print("\n[debug] ── RRF fused order ───────────────────────────────")
        for rank, uid in enumerate(fused_ids):
            print(f"  [{rank+1}] {uid}")

    # ── 4. Fetch fused chunks ───────────────────────────────────────────────
    fetched    = collection.get(ids=fused_ids, include=["documents", "metadatas"])
    id_to_doc  = dict(zip(fetched["ids"], fetched["documents"]))
    id_to_meta = dict(zip(fetched["ids"], fetched["metadatas"]))

    # ── 5. Build context with breadcrumb headers ────────────────────────────
    sections = []
    for uid in fused_ids:
        if uid not in id_to_doc:
            continue
        meta   = id_to_meta[uid]
        crumb  = meta.get("breadcrumb") or meta.get("section") or ""
        header = (
            f"[{meta['spec_id']} — {meta.get('spec_title', '')} | "
            f"{crumb} | chunk {meta.get('chunk', '?')}]"
        )
        sections.append(f"{header}\n{id_to_doc[uid]}")
    context = "\n\n---\n\n".join(sections)

    # ── 6. LLM generation ──────────────────────────────────────────────────
    system_prompt = (
        "You are a precise technical assistant specializing in 3GPP wireless standards. "
        "Answer ONLY using the provided specification context. "
        "You may cite a spec ID (e.g. TS 38.331) or a section path only if that exact "
        "identifier or breadcrumb appears verbatim in the retrieved context headers. "
        "Do not invent, infer, or guess clause numbers or section headings that are not "
        "present in the context. "
        "If the context does not contain enough information to answer the question, "
        "say so explicitly rather than extrapolating from general knowledge."
    )
    user_prompt = f"Context from 3GPP specifications:\n\n{context}\n\nQuestion: {clean_query}"

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

    # ── 7. Source citations ─────────────────────────────────────────────────
    print("\n\nSources:")
    print("─" * 60)
    seen: set = set()
    for uid in fused_ids:
        if uid not in id_to_meta:
            continue
        meta  = id_to_meta[uid]
        crumb = meta.get("breadcrumb") or meta.get("section") or ""
        ctype = meta.get("content_type", "")
        label = (
            f"  • {meta['spec_id']} — {meta.get('spec_title', '')}  |  "
            f"{crumb}  (chunk {meta.get('chunk', '?')}, type={ctype})"
        )
        if label not in seen:
            print(label)
            seen.add(label)
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build and return the argparse CLI parser."""
    p = argparse.ArgumentParser(
        description="3GPP RAG — query indexed standard specifications with hybrid BM25 + vector retrieval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Arguments
---------
  (none)              Interactive Q&A mode.  Type a question at the prompt;
                      the system expands acronyms, runs BM25 + vector retrieval,
                      fuses results with RRF, and streams an LLM answer with
                      source citations.  Type 'exit' or Ctrl-C to quit.

  --stats             Print a summary of the current index and exit:
                      total chunk count, distinct specs and releases, content
                      type breakdown, and the embed model / chunk settings used
                      at ingest time.  Useful to verify the index is up to date
                      before a query session.

  --debug             Enable retrieval trace mode.  Before each LLM answer the
                      system prints: BM25 top-K with scores and text previews,
                      vector top-K with cosine distances and breadcrumbs, and
                      the RRF-fused ordering.  Use this to diagnose why a query
                      is returning unexpected chunks.

  --filter-type TYPE  Restrict both BM25 and vector retrieval to chunks tagged
                      with the given content_type.  Choices:
                        procedure  — narrative specification text (the majority
                                     of spec content; use for conceptual questions
                                     about procedures, timers, conditions)
                        table      — IE presence / range / description tables
                                     (use for questions about field values or
                                     specific IE parameters)
                        asn1       — ASN.1 type definitions and value assignments
                                     (use when looking for formal message
                                     structure, e.g. SEQUENCE fields)

Spec-scoped queries
-------------------
  Restrict retrieval to a single spec by prefixing the question:
    [TS 38.331]: What triggers RRC re-establishment?
  The prefix is parsed and passed as a metadata filter to both BM25 and Chroma.
  The spec_id must match the format stored in the index (e.g. TS 38.331).
  Use --stats to see which spec IDs are indexed.

Query expansion
---------------
  The query is automatically expanded with acronym full-forms and mapped to the
  3GPP internal identifiers that appear verbatim in procedure text.  Examples:
    MIB       → appends "Master Information Block"
    handover  → appends "reconfigurationWithSync"
    HO        → appends "Handover"
  This bridges the gap between natural-language questions and spec terminology
  for both BM25 token matching and embedding similarity.

Typical workflows
-----------------
  General question:
    python query.py
    Query: What is the purpose of SSB?

  Conceptual comparison (filter to prose only):
    python query.py --filter-type procedure
    Query: What is the difference between handover and RRC re-establishment?

  Look up a specific IE field:
    python query.py --filter-type table
    Query: What are the valid values of subcarrierSpacing in SCS-SpecificCarrier?

  Diagnose bad retrieval:
    python query.py --debug
    Query: What triggers beam failure recovery?

  Check index state before querying:
    python query.py --stats
""",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Print index statistics (chunk count, specs, releases, ingest config) and exit.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Show BM25 scores, vector distances, and RRF order before each answer.",
    )
    p.add_argument(
        "--filter-type",
        choices=["procedure", "table", "asn1"],
        metavar="TYPE",
        help="Restrict retrieval to chunks of this type: procedure (prose), table (IE tables), asn1 (formal definitions).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    print("Loading indexes...", end=" ", flush=True)
    bm25, bm25_ids = load_indexes()

    # Fetch metadata for all BM25 IDs so spec/type filtering can be applied
    # on the sparse side in parallel with the Chroma where-clause on the
    # vector side.  Fetched in batches to avoid Chroma payload limits.
    chroma_db_file = CHROMA_DIR / "chroma.sqlite3"
    if not chroma_db_file.exists():
        print("\nERROR: Chroma DB not found. Run  python ingest.py  first.")
        sys.exit(1)

    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    _check_index_config()

    # Fetch metadata for all BM25 IDs (used for BM25-side filtering)
    bm25_metas: List[dict] = []
    batch_size = 500
    for i in range(0, len(bm25_ids), batch_size):
        batch = bm25_ids[i : i + batch_size]
        result = collection.get(ids=batch, include=["metadatas"])
        id_to_meta = dict(zip(result["ids"], result["metadatas"]))
        for uid in batch:
            bm25_metas.append(id_to_meta.get(uid, {}))

    print("ready.\n")

    if args.stats:
        cmd_stats(collection)
        return

    print("=" * 60)
    print("  3GPP Standards RAG & QA System")
    print(f"  Retrieval : BM25-Okapi + nomic-embed-text → RRF (top {FINAL_K})")
    print(f"  Generation: {LLM_MODEL} via Ollama (local)")
    if args.debug:
        print("  Mode      : DEBUG (retrieval trace enabled)")
    if args.filter_type:
        print(f"  Filter    : content_type = {args.filter_type}")
    print("=" * 60)
    print("Spec-scoped queries: prefix with [TS 38.331]: …")
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

        answer(
            query,
            collection,
            bm25,
            bm25_ids,
            bm25_metas,
            debug=args.debug,
            filter_type=args.filter_type,
        )


if __name__ == "__main__":
    main()

