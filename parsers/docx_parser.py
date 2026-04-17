"""
parsers/docx_parser.py — Structure-aware DOCX parser for 3GPP technical specifications.

Design
------
3GPP Word files mix styled headings with numbered Normal paragraphs and embed IE
tables inline between procedure text.  A naive paragraph walk loses section context
and skips tables entirely.  This parser walks the raw XML body in element order so
paragraphs and tables stay interleaved, then groups content into clause sections
delimited by heading detection.

Heading detection uses two paths in priority order:
  1. Word style name (Heading 1-5, Clause Heading, etc.) — used wherever
     rapporteurs applied standard styles.
  2. Numbered-clause regex (e.g. "5.3.1  RRC Connection Setup") — fallback for
     specs that use Normal-styled numbered paragraphs.

Every chunk is prefixed with its full clause breadcrumb so both BM25 token
matching and embedding search have section context even when chunk text alone
is ambiguous.

Content tagging
---------------
Each chunk carries a content_type field set at block granularity:
  - "procedure"  plain paragraph text
  - "table"      IE presence/range/description tables
  - "asn1"       blocks containing ::= (ASN.1 type definitions)

Blocks are split independently so a procedure paragraph that shares a section
with a table is still tagged "procedure", not "table".

Chunk IDs
---------
IDs are SHA-256(spec_id + section_heading + sub_chunk_index), truncated to 16 hex
characters.  Stable provided chunk_size, chunk_overlap, and parser version are
unchanged.  If any of those change, IDs drift and a full re-ingest is required —
incremental mode does not guard against config-driven ID drift.

Known limitation
----------------
3GPP Word specs embed the formal ASN.1 annex as a legacy OLE binary object
(Microsoft_Word_97_-_2003_Document.doc) inside the DOCX ZIP.  python-docx cannot
read OLE2 binary, so formal IE definitions (MasterInformationBlock ::= SEQUENCE
{ ... }) are NOT extracted.  Procedure prose referencing those fields is present.
Use the PDF version of the spec to access formal ASN.1.
"""

import re
import hashlib
from docx import Document
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ── 3GPP version-letter → release-number mapping ──────────────────────────────
# Source: 3GPP TS numbering convention. Letter advances with each new release.
# j=17, i=16, h=15, g=14, f=13, e=12, d=11. Unknown letters → "unknown".
VERSION_TO_RELEASE: Dict[str, str] = {
    "j": "17", "i": "16", "h": "15", "g": "14",
    "f": "13", "e": "12", "d": "11", "c": "10",
}

# ── Word heading style names used in 3GPP specs ───────────────────────────────
# Standard Word heading styles (preferred path). Checked via para.style.name.
HEADING_STYLE_NAMES = {
    "Heading 1", "Heading 2", "Heading 3", "Heading 4", "Heading 5",
    # Some 3GPP rapporteurs use localised or custom variants:
    "heading 1", "heading 2", "heading 3",
    "H1", "H2", "H3", "H4",
    "Clause Heading", "Annex Heading", "AnnexA",
}

# ── Regex fallback: numbered clause lines in Normal style ─────────────────────
# Matches "5.3.1  RRC Connection Setup", "A.1  Scope", "B.2.3  Parameters".
# Requires at least one dot segment (e.g. 5.1, A.2) so bare sentences
# starting with a single digit ("1 The UE shall...") are not false-positives.
# Top-level single-digit clauses ("5  Procedures") are caught by Heading
# style detection (path 1) and do not need the regex fallback.
CLAUSE_NUMBER_RE = re.compile(
    r"^([A-Z]|\d+)(\.\d+){1,4}\s{1,4}\S"
)

# ── Boilerplate patterns to drop before chunking (Change 8) ──────────────────
# These appear on every page of 3GPP Word docs as header/footer artefacts.
BOILERPLATE_RE = re.compile(
    r"^(3GPP\s+TS\s+\d+\.\d+|ETSI|Release\s+\d+|"
    r"Technical\s+Specification|"
    r"3rd\s+Generation\s+Partnership\s+Project|"
    r"Post\s+date|Draft|CR\s+page)",
    re.IGNORECASE,
)

# ── Max breadcrumb prefix length (chars) ─────────────────────────────────────
# Caps the prepended heading path to avoid oversized embeddings.
MAX_BREADCRUMB_LEN = 120


class DocxParser:
    """
    Structure-aware parser for 3GPP DOCX specification files.

    Extracts paragraphs and tables in document order, tracks the current
    clause hierarchy (breadcrumb), filters boilerplate, tags content type,
    and produces overlapping chunks with stable SHA-256 IDs.

    Parameters
    ----------
    chunk_size : int
        Maximum characters per chunk (secondary split inside a section).
    chunk_overlap : int
        Character overlap between consecutive sub-chunks within a section.
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_document(self, file_path: str, spec_id: str = "") -> List[Dict]:
        """
        Full pipeline: parse a DOCX file and return a flat list of chunk dicts.

        Each chunk dict contains:
            text          : chunk content (breadcrumb prepended)
            chunk_id      : stable SHA-256 ID (16 hex chars)
            spec_id       : e.g. "TS 38.331"
            section       : current clause heading breadcrumb
            content_type  : "procedure" | "table" | "asn1"
            chunk_index   : 0-based index within its section
            chunk_length  : character count

        Parameters
        ----------
        file_path : str
            Absolute or relative path to the .docx file.
        spec_id : str
            Spec identifier injected from ingest.py (e.g. "TS 38.331").
            Used for deterministic ID generation.
        """
        doc = Document(file_path)
        sections = self._extract_sections(doc)
        chunks: List[Dict] = []
        for section in sections:
            chunks.extend(self._chunk_section(section, spec_id))
        return chunks

    @staticmethod
    def parse_release_from_filename(filename: str) -> str:
        """
        Extract the 3GPP release number from a spec filename.

        The 3GPP naming convention embeds a version letter immediately after
        the spec number and a dash, e.g. "38331-j20" → letter "j" → Release 17.

        Parameters
        ----------
        filename : str
            Filename such as "38331-j20 NR Radio Resource Control.docx".

        Returns
        -------
        str
            Release number string (e.g. "17") or "unknown" if not recognised.
        """
        m = re.search(r"-([a-z])\d+", Path(filename).stem, re.IGNORECASE)
        if m:
            letter = m.group(1).lower()
            return VERSION_TO_RELEASE.get(letter, "unknown")
        return "unknown"

    # ── Heading detection ──────────────────────────────────────────────────────

    @staticmethod
    def _is_heading(para) -> Tuple[bool, str]:
        """
        Determine whether a python-docx Paragraph is a section heading.

        Uses a three-path detection strategy with logging counters stored
        on the class (accessible via _heading_stats after a parse run):

          Path 1 (preferred):  Word style name in HEADING_STYLE_NAMES
          Path 2 (fallback):   Text matches CLAUSE_NUMBER_RE (numbered clause)
          Path 3 (default):    Not a heading

        Returns
        -------
        (is_heading: bool, source: str)
            source is "style" | "regex" | "none"
        """
        style_name = para.style.name if para.style else ""
        if style_name in HEADING_STYLE_NAMES:
            return True, "style"
        text = para.text.strip()
        if text and CLAUSE_NUMBER_RE.match(text):
            return True, "regex"
        return False, "none"

    # ── Document extraction ────────────────────────────────────────────────────

    def _extract_sections(self, doc: Document) -> List[Dict]:
        """
        Walk the document body XML in element order and group content into
        sections delimited by heading paragraphs.

        Each section dict:
            heading       : str  — clause title (e.g. "5.3.1 RRC Connection Setup")
            breadcrumb    : str  — full path e.g. "5.3 > 5.3.1 RRC Connection Setup"
            blocks        : List[Dict]  — content blocks with text + content_type

        Paragraphs and tables are processed in document order (not separately),
        preserving the relationship between procedure text and IE tables.

        Boilerplate lines are filtered before being added to blocks.

        Heading detection stats are accumulated in self._heading_stats for
        post-parse diagnostics.
        """
        self._heading_stats = {"style": 0, "regex": 0, "content_paragraphs": 0}

        # Heading stack: list of (level, text) pairs for breadcrumb building.
        # Level is inferred from style name or regex depth.
        heading_stack: List[Tuple[int, str]] = []
        current_heading = "Preamble"
        current_blocks: List[Dict] = []
        sections: List[Dict] = []

        def _flush(heading: str, breadcrumb: str, blocks: List[Dict]) -> None:
            """Save current section if it has non-empty content."""
            non_empty = [b for b in blocks if b["text"].strip()]
            if non_empty:
                sections.append({
                    "heading": heading,
                    "breadcrumb": breadcrumb,
                    "blocks": non_empty,
                })

        def _heading_level(para) -> int:
            """Infer numeric heading level (1–5) from style or regex depth."""
            style = para.style.name if para.style else ""
            for i in range(1, 6):
                if str(i) in style:
                    return i
            # Regex path: count dot-separated segments
            m = CLAUSE_NUMBER_RE.match(para.text.strip() if para.text else "")
            if m:
                return min(para.text.strip().count(".") + 1, 5)
            return 1

        def _build_breadcrumb() -> str:
            """Build a breadcrumb string from heading_stack, capped at MAX_BREADCRUMB_LEN."""
            if not heading_stack:
                return current_heading
            path = " > ".join(text for _, text in heading_stack[-3:])
            return path[:MAX_BREADCRUMB_LEN]

        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                # ── Extract raw text from all w:t nodes ───────────────────────
                raw_text = "".join(
                    n.text for n in element.iter()
                    if n.tag.endswith("}t") and n.text
                ).strip()

                if not raw_text:
                    continue

                # ── Boilerplate filter (Change 8) ─────────────────────────────
                if BOILERPLATE_RE.match(raw_text) or len(raw_text) < 4:
                    continue

                # ── Heading detection ─────────────────────────────────────────
                # Wrap element in a python-docx Paragraph to access .style
                from docx.text.paragraph import Paragraph as DocxParagraph
                para_obj = DocxParagraph(element, doc)
                is_hdg, hdg_source = self._is_heading(para_obj)

                if is_hdg:
                    # Flush previous section
                    _flush(current_heading, _build_breadcrumb(), current_blocks)
                    current_blocks = []

                    # Update heading stack
                    level = _heading_level(para_obj)
                    # Pop stack entries at same or deeper level
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, raw_text))
                    current_heading = raw_text
                    self._heading_stats[hdg_source] += 1
                else:
                    # Determine content type
                    content_type = "asn1" if "::=" in raw_text else "procedure"
                    current_blocks.append({"text": raw_text, "content_type": content_type})
                    self._heading_stats["content_paragraphs"] += 1

            elif tag == "tbl":
                # ── Table extraction: row × cell → " | " joined rows ──────────
                # Preserves IE field description tables (Field | Presence | Range | Desc)
                # and any ASN.1 blocks stored as single-cell tables.
                rows_text: List[str] = []
                for tr in element:
                    if not tr.tag.endswith("}tr"):
                        continue
                    cells: List[str] = []
                    for tc in tr:
                        if not tc.tag.endswith("}tc"):
                            continue
                        cell_text = "".join(
                            n.text for n in tc.iter()
                            if n.tag.endswith("}t") and n.text
                        ).strip()
                        if cell_text:
                            cells.append(cell_text)
                    if cells:
                        rows_text.append(" | ".join(cells))

                if rows_text:
                    table_text = "\n".join(rows_text)
                    content_type = "asn1" if "::=" in table_text else "table"
                    current_blocks.append({"text": table_text, "content_type": content_type})

        # Flush final section
        _flush(current_heading, _build_breadcrumb(), current_blocks)
        return sections

    # ── Section → chunks ──────────────────────────────────────────────────────

    def _chunk_section(self, section: Dict, spec_id: str) -> List[Dict]:
        """
        Split a single section's blocks into overlapping sub-chunks.

        Each block is split independently so its content_type is preserved
        per sub-chunk rather than collapsing to a single section-level type.
        For example, a narrative procedure paragraph that shares a section
        with an IE table will still be tagged "procedure", not "table"
        (review issue 4 fix).

        The section breadcrumb is prepended to every sub-chunk so the LLM
        always has clause context, even when the section is split across
        multiple chunks.

        Chunk IDs are deterministic: sha256(spec_id + heading + sub_index)
        so they remain stable across re-ingests (prerequisite for incremental
        ingest in Change 12).

        Parameters
        ----------
        section : dict
            Output of _extract_sections — has "heading", "breadcrumb", "blocks".
        spec_id : str
            E.g. "TS 38.331", injected from ingest.py.

        Returns
        -------
        List of chunk dicts ready for Chroma + BM25 ingestion.
        """
        breadcrumb = section["breadcrumb"]
        heading    = section["heading"]
        blocks     = section["blocks"]

        # Split each block independently to carry its content_type into every
        # sub-chunk it produces. Overlap is preserved within each block.
        flat: List[Tuple[str, str]] = []
        for block in blocks:
            for part in self._split_text(block["text"]):
                if part.strip():
                    flat.append((part, block["content_type"]))

        chunks: List[Dict] = []
        breadcrumb_prefix = f"[{breadcrumb[:MAX_BREADCRUMB_LEN]}]\n"

        for sub_idx, (sub_text, content_type) in enumerate(flat):
            chunk_text = breadcrumb_prefix + sub_text

            # Deterministic ID: stable across re-ingests (Change 9)
            uid = hashlib.sha256(
                f"{spec_id}\0{heading}\0{sub_idx}".encode()
            ).hexdigest()[:16]

            chunks.append({
                "text":         chunk_text,
                "chunk_id":     uid,
                "spec_id":      spec_id,
                "section":      heading,
                "breadcrumb":   breadcrumb,
                "content_type": content_type,
                "chunk_index":  sub_idx,
                "chunk_length": len(chunk_text),
            })

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Split a text block into overlapping sub-chunks bounded by self.chunk_size.

        Boundary preference order (within the backtrack window):
          1. Paragraph double-newline
          2. Sentence-ending punctuation (. ! ?)
          3. Whitespace (word boundary)
          4. Hard cut at chunk_size

        Parameters
        ----------
        text : str
            Raw section text (no breadcrumb prefix yet).

        Returns
        -------
        List of text strings, each ≤ chunk_size characters.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                # 1. Paragraph break
                found = False
                for i in range(end, max(start + self.chunk_size - 200, start), -1):
                    if text[i : i + 2] == "\n\n":
                        end = i + 2
                        found = True
                        break
                # 2. Sentence ending
                if not found:
                    for i in range(end, max(start + self.chunk_size - 100, start), -1):
                        if text[i] in (".", "!", "?"):
                            end = i + 1
                            found = True
                            break
                # 3. Word boundary
                if not found:
                    for i in range(end, max(start + self.chunk_size - 50, start), -1):
                        if text[i].isspace():
                            end = i
                            break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap if end < len(text) else end

        return chunks

    def heading_stats(self) -> Dict[str, int]:
        """
        Return heading detection path counts from the last parse run.

        Useful for validating that heading styles in a spec file are being
        detected correctly vs falling back to regex.

        Returns
        -------
        dict with keys:
            "style"              — headings detected via Word style name
            "regex"              — headings detected via CLAUSE_NUMBER_RE fallback
            "content_paragraphs" — non-heading content blocks processed
        """
        return getattr(self, "_heading_stats", {"style": 0, "regex": 0, "content_paragraphs": 0})
