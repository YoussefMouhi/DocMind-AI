from pathlib import Path

import pymupdf
from docx import Document


def load_pdf(file_path):
    """Extract text from a PDF, preserving page numbers."""
    pages = []

    with pymupdf.open(file_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "source": Path(file_path).name,
                    "page": page_number,
                    "text": text,
                })

    return pages


def load_docx(file_path):
    """Extract paragraphs from a Word document."""
    document = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return [{
        "source": Path(file_path).name,
        "page": None,
        "text": text,
    }] if text else []


def load_txt(file_path):
    """Extract text from a TXT file."""
    text = Path(file_path).read_text(
        encoding="utf-8-sig"
    ).strip()

    return [{
        "source": Path(file_path).name,
        "page": None,
        "text": text,
    }] if text else []


def load_document(file_path):
    """Select the appropriate loader by file extension."""
    file_path = Path(file_path)

    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt,
    }

    extension = file_path.suffix.lower()

    if extension not in loaders:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    return loaders[extension](file_path)
