# 3GPP RAG — Standards Q&A with Hybrid Retrieval

A local Retrieval-Augmented Generation (RAG) system for querying 3GPP technical specifications. It combines **BM25 sparse retrieval** and **ChromaDB vector search**, fused with **Reciprocal Rank Fusion (RRF)**, and answered by a local LLM via [Ollama](https://ollama.com).

## How it works

```
3GPP .docx/.pdf files
        │
        ▼
   ingest.py  ──►  ChromaDB (vector index)
              ──►  BM25 index (pickle)
                        │
                        ▼
   query.py   ──►  Hybrid retrieval (BM25 + vector, RRF fusion)
              ──►  Ollama LLM (llama3.2:3b)
              ──►  Answer
```

## Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com)** installed and running locally
3. Pull the required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2:3b
   ```

## Setup

```bash
# Clone the repo
git clone https://github.com/waveweaver/3gpp-rag.git
cd 3gpp-rag

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Adding 3GPP Specs

Place your 3GPP specification files (`.docx` or `.pdf`) in `data/specs/`.

Specs can be downloaded from the [3GPP FTP server](https://www.3gpp.org/ftp/Specs/archive/).

Files should follow the standard 3GPP naming convention, e.g.:
```
38331-j20 NR Radio Resource Control (RRC); Protocol specification.docx
```

## Usage

### Step 1 — Ingest specs (run once, or when specs are updated)

```bash
python ingest.py
```

This parses all files in `data/specs/`, chunks the text, embeds each chunk with `nomic-embed-text`, and stores everything in `chroma_db/` and `index/`.

### Step 2 — Query interactively

```bash
python query.py
```

Type a question at the prompt and the system will retrieve the most relevant chunks and generate an answer using the local LLM.

```
Ask a question (or 'exit'): What are the RRC states in NR?
```

## Project Structure

```
3gpp-rag/
├── ingest.py           # Parses specs and builds the hybrid index
├── query.py            # Interactive Q&A interface
├── requirements.txt
├── parsers/
│   ├── docx_parser.py  # DOCX chunker with metadata extraction
│   └── pdf_parser.py   # PDF chunker using LangChain text splitter
├── data/
│   └── specs/          # Place 3GPP .docx/.pdf files here (git-ignored)
├── chroma_db/          # ChromaDB vector store (generated, git-ignored)
└── index/              # BM25 pickle files (generated, git-ignored)
```

## Configuration

Key constants at the top of each script:

| Constant | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `LLM_MODEL` | `llama3.2:3b` | Ollama LLM for answer generation |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `K_EACH` | `6` | Candidates fetched per retriever |
| `FINAL_K` | `5` | Chunks sent to the LLM after RRF fusion |
