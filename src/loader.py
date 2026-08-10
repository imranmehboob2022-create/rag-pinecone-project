"""
loader.py
---------
Responsible for ONE thing: turning an uploaded PDF into clean,
page-tagged text. No chunking, no embeddings here (separation of
concerns).
"""

import re
import io
from dataclasses import dataclass
from typing import List

import pdfplumber


class InvalidPDFError(Exception):
    """Raised when the uploaded file is not a readable / valid PDF."""
    pass


class PDFTooLargeError(Exception):
    """Raised when the uploaded file exceeds the configured size limit."""
    pass


@dataclass
class PageText:
    page_number: int      # 1-indexed, human-friendly
    text: str
    document_name: str


def _clean_text(raw: str) -> str:
    """Remove common PDF extraction artifacts: excessive whitespace,
    hyphenation line-breaks, repeated headers/footers whitespace, etc."""
    if not raw:
        return ""

    text = raw

    # Fix words broken across lines with a hyphen: "exam-\nple" -> "example"
    text = re.sub(r"-\n(?=\w)", "", text)

    # Collapse multiple newlines into a single space (keep paragraph feel)
    text = re.sub(r"\n+", " ", text)

    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip stray page-number-only artifacts like "12" on their own if very short
    text = text.strip()

    return text


def validate_pdf_bytes(file_bytes: bytes, filename: str, max_size_mb: int) -> None:
    """Basic validation before we even try to parse the file."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise PDFTooLargeError(
            f"'{filename}' is {size_mb:.1f} MB, which exceeds the {max_size_mb} MB limit."
        )
    if not file_bytes.startswith(b"%PDF"):
        raise InvalidPDFError(f"'{filename}' does not look like a valid PDF file.")


def extract_text_by_page(file_bytes: bytes, filename: str, max_size_mb: int = 20) -> List[PageText]:
    """
    Extract text from a PDF, page by page, with basic cleaning applied.

    Returns a list of PageText objects. Raises InvalidPDFError /
    PDFTooLargeError on bad input so the UI layer can show a friendly
    error message instead of crashing.
    """
    validate_pdf_bytes(file_bytes, filename, max_size_mb)

    pages: List[PageText] = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if len(pdf.pages) == 0:
                raise InvalidPDFError(f"'{filename}' contains no pages.")

            for i, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""
                cleaned = _clean_text(raw_text)
                if cleaned:
                    pages.append(
                        PageText(page_number=i, text=cleaned, document_name=filename)
                    )
    except InvalidPDFError:
        raise
    except Exception as exc:
        raise InvalidPDFError(f"Could not parse '{filename}': {exc}") from exc

    if not pages:
        raise InvalidPDFError(
            f"'{filename}' produced no extractable text (it may be a scanned image PDF)."
        )

    return pages
