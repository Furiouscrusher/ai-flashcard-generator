"""
test_rag_pipeline.py

Smoke test demonstrating Member 2's RAG Pipeline and AI Flashcard Generation.
1. Ingests sample study document (sample_docs/unit3.pdf) using Member 1's processor.
2. Indexes chunks and embeddings into ChromaDB.
3. Performs semantic search on a study topic.
4. Generates structured, validated flashcards with source citations and difficulty levels.
5. Evaluates relevance against source context and exports cards to sample_flashcards.json.
"""

import json
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modules.document_processor import process_document
from modules.models import GenerationRequest
from modules.vector_store import ChromaVectorStore
from modules.llm_client import get_llm_client
from modules.flashcard_generator import generate_flashcards


def main():
    print("=" * 60)
    print("  AI Flashcard Generator Using RAG - Member 2 Demonstration")
    print("=" * 60)

    # 1. Process Document using Member 1's module
    pdf_path = Path("sample_docs/unit3.pdf")
    if not pdf_path.exists():
        print(f"Error: Sample PDF not found at {pdf_path}")
        return

    print(f"\n[1] Processing document: {pdf_path.name}...")
    file_bytes = pdf_path.read_bytes()
    chunks = process_document(
        file_bytes=file_bytes,
        filename=pdf_path.name,
        chunk_size=1000,
        overlap=150,
    )
    print(f"    [+] Extracted and prepared {len(chunks)} text chunks.")

    # 2. Vector Store Ingestion (ChromaDB)
    print("\n[2] Initializing ChromaDB vector store...")
    vector_store = ChromaVectorStore(
        collection_name="demo_study_flashcards",
        use_in_memory=True,  # In-memory for fast and clean demonstration
    )
    added_count = vector_store.add_chunks(chunks)
    print(f"    [+] Successfully indexed {added_count} chunks in ChromaDB.")

    # 3. Semantic Retrieval Test
    query_topic = "Cell Structure and Mitochondria"
    print(f"\n[3] Performing semantic search for: '{query_topic}'...")
    retrieved = vector_store.similarity_search(query=query_topic, top_k=3)
    print(f"    [+] Retrieved top {len(retrieved)} relevant chunks:")
    for idx, r in enumerate(retrieved, start=1):
        citation = r.format_source_citation()
        snippet = r.text[:120].replace("\n", " ")
        print(f"      {idx}. Score: {r.score:.3f} | {citation} | \"{snippet}...\"")

    # 4. Flashcard Generation Pipeline
    print(f"\n[4] Generating flashcards for topic: '{query_topic}'...")
    request = GenerationRequest(
        topic=query_topic,
        num_flashcards=4,
        difficulty="Mixed",
        top_k=4,
    )

    llm = get_llm_client()
    print(f"    [+] Using LLM Client: {llm.__class__.__name__}")

    flashcard_set = generate_flashcards(
        request=request,
        vector_store=vector_store,
        llm_client=llm,
    )

    print(f"\n[5] Generated {len(flashcard_set.flashcards)} validated flashcards:\n")
    for idx, card in enumerate(flashcard_set.flashcards, start=1):
        print(f"  --- Flashcard #{idx} [{card.difficulty.upper()}] " + "-" * 35)
        print(f"  Question:   {card.question}")
        print(f"  Answer:     {card.answer}")
        print(f"  Topic:      {card.topic}")
        print(f"  Source:     {card.source}")
        print(f"  Chunk IDs:  {', '.join(card.source_chunk_ids)}")
        print("  " + "-" * 55 + "\n")

    # 5. Relevance Evaluation Report
    rel_report = flashcard_set.metadata.get("relevance_report", {})
    print("[6] Content Relevance & Grounding Evaluation:")
    print(f"    - Overall Grounded:  {rel_report.get('is_relevant', False)}")
    print(f"    - Avg Token Overlap: {rel_report.get('average_relevance', 0.0):.1%}")
    print(f"    - Grounded Cards:    {rel_report.get('grounded_count', 0)} / {rel_report.get('total_cards', 0)}")

    # 6. Export to JSON (For Member 3 Integration)
    export_path = Path("sample_flashcards.json")
    export_data = {
        "topic": flashcard_set.topic,
        "difficulty": flashcard_set.difficulty_target,
        "flashcards": flashcard_set.to_dict_list(),
        "metadata": flashcard_set.metadata,
    }
    export_path.write_text(json.dumps(export_data, indent=2), encoding="utf-8")
    print(f"\n[7] Exported flashcards successfully to: {export_path.resolve()}")
    print("\n[OK] Member 2 demonstration complete!\n")


if __name__ == "__main__":
    main()
