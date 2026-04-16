from docx import Document
from pathlib import Path
from typing import List, Dict
import hashlib


class DocxParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_docx(self, file_path: str) -> Dict:
        """Parse a DOCX file and extract content with metadata."""
        path = Path(file_path)
        doc = Document(file_path)

        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        content = "\n\n".join(paragraphs)

        core_props = doc.core_properties
        metadata = {
            "filename": path.name,
            "file_path": str(path.absolute()),
            "title": core_props.title or "Untitled",
            "author": core_props.author or "Unknown",
            "created": str(core_props.created) if core_props.created else None,
            "modified": str(core_props.modified) if core_props.modified else None,
        }

        return {"content": content, "paragraphs": paragraphs, "metadata": metadata}

    def chunk_text(self, text: str) -> List[Dict]:
        """Split text into overlapping chunks, breaking at natural boundaries."""
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                # Prefer paragraph breaks
                for i in range(end, max(start + self.chunk_size - 200, start), -1):
                    if i < len(text) and text[i : i + 2] == "\n\n":
                        end = i + 2
                        break
                else:
                    # Then sentence endings
                    for i in range(end, max(start + self.chunk_size - 100, start), -1):
                        if i < len(text) and text[i] in (".", "!", "?"):
                            end = i + 1
                            break
                    else:
                        # Finally word boundaries
                        for i in range(end, max(start + self.chunk_size - 50, start), -1):
                            if i < len(text) and text[i].isspace():
                                end = i
                                break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "start_char": start,
                    "end_char": end,
                    "chunk_length": len(chunk_text),
                })
                chunk_id += 1

            start = end - self.chunk_overlap if end < len(text) else end

        return chunks

    def process_document(self, file_path: str) -> List[Dict]:
        """Full pipeline: parse DOCX and return chunks with document metadata."""
        doc_data = self.parse_docx(file_path)
        chunks = self.chunk_text(doc_data["content"])
        for chunk in chunks:
            chunk["document_metadata"] = doc_data["metadata"]
        return chunks

    def _generate_doc_id(self, file_path: str) -> str:
        return hashlib.md5(file_path.encode()).hexdigest()
