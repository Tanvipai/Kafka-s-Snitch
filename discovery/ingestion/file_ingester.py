import hashlib
import logging
from pathlib import Path

import chardet
import magic
import fitz  
from docx import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SUPPORTED_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "text",
    "text/csv": "text",
    "text/x-python": "text",
    "text/x-script.python": "text",  
    "application/octet-stream": "unknown",
}


class FileIngester:

    def ingest(self, file_path: str) -> dict:
        path = Path(file_path)

        result = {
            "file_path": str(path),
            "file_name": path.name,
            "file_type": None,
            "encoding": None,
            "text": None,
            "char_count": 0,
            "file_hash": None,
            "error": None,
        }

        if not path.exists():
            result["error"] = "File not found"
            return result

        try:
            result["file_hash"] = self._hash_file(path)
            mime_type = magic.from_file(str(path), mime=True)
            result["file_type"] = SUPPORTED_TYPES.get(mime_type, "unsupported")

            if result["file_type"] == "pdf":
                result["text"] = self._extract_pdf(path)
            elif result["file_type"] == "docx":
                result["text"] = self._extract_docx(path)
            elif result["file_type"] == "text":
                encoding = self._detect_encoding(path)
                result["encoding"] = encoding
                result["text"] = path.read_text(encoding=encoding, errors="replace")
            else:
                result["error"] = f"Unsupported file type: {mime_type}"
                return result

            result["char_count"] = len(result["text"]) if result["text"] else 0

        except Exception as e:
            logger.error(f"Failed to ingest {path.name}: {e}")
            result["error"] = str(e)

        return result

    def _hash_file(self, path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _detect_encoding(self, path: Path) -> str:
        raw = path.read_bytes()
        detected = chardet.detect(raw)
        return detected.get("encoding") or "utf-8"

    def _extract_pdf(self, path: Path) -> str:
        text_parts = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        return "\n".join(text_parts)

    def _extract_docx(self, path: Path) -> str:
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())