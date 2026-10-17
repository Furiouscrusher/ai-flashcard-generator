
from pathlib import Path
from modules.document_processor import (
    process_document,
    DocumentProcessingError,
)

file_path = Path("sample_docs/unit3.pdf")

try:
    file_bytes = file_path.read_bytes()

    chunks = process_document(
        file_bytes=file_bytes,
        filename=file_path.name,
        chunk_size=1000,
        overlap=150,
    )

    print(f"Total chunks generated: {len(chunks)}")

    for chunk in chunks[:3]:
        print("\n" + "=" * 50)
        print("Chunk ID:", chunk["chunk_id"])
        print("Metadata:", chunk["metadata"])
        print("Text:", chunk["text"][:300])

except (OSError, DocumentProcessingError) as error:
    print("Processing failed:", error)
