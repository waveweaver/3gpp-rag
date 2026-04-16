from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFParser:
    """Parse PDF files and prepare them for RAG ingestion."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from a PDF file."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text = ""
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text

    def chunk_text(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Parse a PDF and return chunked text with metadata."""
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        return [
            {
                "chunk_id": idx,
                "text": chunk,
                "source": pdf_path,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks),
            }
            for idx, chunk in enumerate(chunks)
        ]
