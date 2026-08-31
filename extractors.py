"""
extractors.py
Turn uploaded report files (PDF, DOCX, TXT) into clean plain text that the
summarizer can consume.

Day 2 goal: extract_text(file_path_or_bytes) -> str, working for all three
formats, with basic cleanup of page numbers / headers / footers / stray
whitespace that commonly pollute extracted report text.
"""

import io
import re
from pathlib import Path

from pypdf import PdfReader
from docx import Document


# ---------------------------------------------------------------------------
# Format-specific extraction
# ---------------------------------------------------------------------------

def extract_pdf(file) -> str:
    """
    Extract text from a PDF.

    Args:
        file: a file path (str/Path), an open binary file object, or bytes.

    Returns:
        Extracted text with a blank line between pages.
    """
    reader = PdfReader(file)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n\n".join(pages_text)


def extract_docx(file) -> str:
    """
    Extract text from a Word document.

    Args:
        file: a file path (str/Path), an open binary file object, or bytes.

    Returns:
        Extracted text, one paragraph per line. Also pulls text out of any
        tables in the document, since reports often put key figures there.
    """
    document = Document(file)

    parts = []
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            parts.append(paragraph.text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts)


def extract_txt(file) -> str:
    """
    Extract text from a plain text file.

    Args:
        file: a file path (str/Path), an open binary/text file object, or
            bytes/str.
    """
    if isinstance(file, (str, Path)) and not isinstance(file, bytes):
        # could be a path, or could already be the text content itself --
        # only treat it as a path if it actually points to an existing file
        path = Path(file)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
        return str(file)

    if isinstance(file, bytes):
        return file.decode("utf-8", errors="ignore")

    if hasattr(file, "read"):
        content = file.read()
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        return content

    return str(file)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

# Matches lines that are *just* a page number, e.g. "Page 4", "4", "- 4 -"
_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s*)?[-–—]?\s*\d{1,4}\s*[-–—]?\s*$", re.IGNORECASE)

# Common report boilerplate lines worth stripping when they appear alone
_BOILERPLATE_LINE = re.compile(
    r"^\s*(confidential|internal use only|draft|do not distribute)\s*[-–—:]*\s*(internal use only)?\s*$",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """
    Strip common extraction noise: standalone page-number lines, repeated
    "CONFIDENTIAL" / "INTERNAL USE ONLY" headers-footers, excess blank
    lines, and broken mid-sentence line breaks.
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if _PAGE_NUMBER_LINE.match(stripped):
            continue
        if _BOILERPLATE_LINE.match(stripped):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # collapse 3+ blank lines down to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # collapse runs of spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    # rejoin a word that got hyphen-broken across a line ("govern-\nance")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".txt": extract_txt,
}


def extract_text(file, filename: str | None = None) -> str:
    """
    Route a file to the right extractor based on its extension, then clean
    the result.

    Args:
        file: a file path (str/Path) or an open/uploaded file object
            (e.g. Streamlit's UploadedFile).
        filename: required when `file` is not a path itself (e.g. a
            Streamlit UploadedFile) -- used only to determine the
            extension. If omitted, `file` is assumed to be a path.

    Returns:
        Cleaned extracted text.

    Raises:
        ValueError: if the file extension isn't supported.
    """
    name = filename or str(file)
    ext = Path(name).suffix.lower()

    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: "
            f"{', '.join(EXTRACTORS.keys())}"
        )

    raw_text = extractor(file)
    return clean_text(raw_text)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extractors.py <path-to-file>")
        sys.exit(1)

    path = sys.argv[1]
    text = extract_text(path)
    print(f"--- Extracted {len(text)} characters from {path} ---\n")
    print(text[:1000])
    if len(text) > 1000:
        print("\n... (truncated)")
