
"""
document_processor.py

Member 1: Document Processing and Data Preparation
Supports PDF and TXT files.
"""

import hashlib
import re
from pathlib import Path

import pymupdf


class DocumentProcessingError(Exception):
    """Raised when a document cannot be processed."""


def clean_text(text: str) -> str:
    """Clean extracted text while preserving meaningful symbols."""

    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[str]:
    """Split text into word-safe chunks with approximate overlap."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    text = text.strip()

    if not text:
        return []

    words = list(re.finditer(r"\S+", text))

    if not words:
        return []

    chunks = []
    start_index = 0

    while start_index < len(words):
        start_char = words[start_index].start()
        target_end = start_char + chunk_size

        end_index = start_index

        while (
            end_index < len(words)
            and words[end_index].end() <= target_end
        ):
            end_index += 1

        # Include at least one word, even if it exceeds chunk_size.
        if end_index == start_index:
            end_index += 1

        # Prefer a word boundary near the target chunk size.
        if end_index < len(words):
            minimum_end = start_char + int(chunk_size * 0.70)

            while (
                end_index > start_index + 1
                and words[end_index - 1].end() > target_end
            ):
                end_index -= 1

            if (
                end_index < len(words)
                and words[end_index].start() < target_end
                and words[end_index].end() > minimum_end
            ):
                end_index += 1

        end_char = words[end_index - 1].end()
        chunks.append(text[start_char:end_char].strip())

        if end_index >= len(words):
            break

        # Retain approximate overlap between consecutive chunks.
        next_start_char = max(start_char, end_char - overlap)
        next_index = end_index - 1

        while (
            next_index > start_index
            and words[next_index].start() > next_start_char
        ):
            next_index -= 1

        start_index = max(start_index + 1, next_index)

    return chunks


def process_document(
    file_bytes: bytes,
    filename: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    """
    Process an uploaded PDF or TXT document.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename, including its extension.
        chunk_size: Approximate maximum chunk length in characters.
        overlap: Approximate overlap between consecutive chunks.

    Returns:
        A list of dictionaries containing:
        - chunk_id
        - text
        - metadata: document, page, document_id

    Raises:
        DocumentProcessingError for invalid or unreadable documents.
        ValueError for invalid chunk settings.
    """

    if not isinstance(file_bytes, bytes) or not file_bytes:
        raise DocumentProcessingError("The uploaded file is empty.")

    if not isinstance(filename, str) or not filename.strip():
        raise DocumentProcessingError("A valid filename is required.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    extension = Path(filename).suffix.lower()
    extracted_sections = []

    # Stable identifier based on the document's content.
    document_id = hashlib.sha256(file_bytes).hexdigest()[:12]

    try:
        if extension == ".pdf":

            with pymupdf.open(
                stream=file_bytes,
                filetype="pdf",
            ) as pdf:

                if pdf.page_count == 0:
                    raise DocumentProcessingError(
                        "The PDF contains no pages."
                    )

                for page_index, page in enumerate(pdf, start=1):
                    page_text = clean_text(page.get_text("text"))

                    if page_text:
                        extracted_sections.append(
                            (page_text, page_index)
                        )

        elif extension == ".txt":

            try:
                raw_text = file_bytes.decode("utf-8-sig")

            except UnicodeDecodeError:
                try:
                    raw_text = file_bytes.decode("cp1252")

                except UnicodeDecodeError as exc:
                    raise DocumentProcessingError(
                        "Unable to decode the TXT file. "
                        "Save it as UTF-8 and try again."
                    ) from exc

            text = clean_text(raw_text)

            if text:
                extracted_sections.append((text, None))

        else:
            raise DocumentProcessingError(
                "Unsupported file type. Upload a PDF or TXT file."
            )

    except DocumentProcessingError:
        raise

    except Exception as exc:
        raise DocumentProcessingError(
            "The document could not be read. "
            "Check whether it is corrupted or password-protected."
        ) from exc

    if not extracted_sections:
        raise DocumentProcessingError(
            "No extractable text was found. "
            "The PDF may contain scanned images and require OCR."
        )

    results = []
    document_stem = Path(filename).stem

    for section_text, page_number in extracted_sections:

        chunks = split_into_chunks(
            section_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_index, chunk in enumerate(chunks, start=1):

            page_label = (
                f"p{page_number}"
                if page_number is not None
                else "txt"
            )

            results.append({
                "chunk_id": (
                    f"{document_stem}_{document_id}_"
                    f"{page_label}_c{chunk_index}"
                ),
                "text": chunk,
                "metadata": {
                    "document": Path(filename).name,
                    "document_id": document_id,
                    "page": page_number,
                },
            })

    if not results:
        raise DocumentProcessingError(
            "No usable text chunks could be generated."
        )

    return results
