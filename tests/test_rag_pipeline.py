"""
test_rag_pipeline.py

Unit tests for Member 2: RAG Pipeline and AI Flashcard Generation.
Tests vector storage, semantic search, prompt generation, Pydantic validation,
deduplication, relevance scoring, and end-to-end pipeline execution.
"""

import unittest
from unittest.mock import MagicMock

from modules.models import (
    Flashcard,
    FlashcardSet,
    GenerationRequest,
    RetrievedChunk,
)
from modules.prompts import build_flashcard_prompt, format_context_chunks
from modules.llm_client import (
    MockLLMClient,
    extract_json_from_response,
)
from modules.vector_store import ChromaVectorStore
from modules.flashcard_generator import (
    FlashcardGenerationError,
    deduplicate_flashcards,
    evaluate_flashcard_relevance,
    generate_flashcards,
    normalize_question,
)


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        """Set up in-memory vector store and sample chunks for testing."""
        self.vector_store = ChromaVectorStore(
            collection_name="test_collection",
            use_in_memory=True,
        )

        self.sample_chunks = [
            {
                "chunk_id": "os_notes_1_p1_c1",
                "text": "An operating system manages computer hardware and system resources such as memory, CPU, and storage.",
                "metadata": {
                    "document": "os_notes.pdf",
                    "document_id": "doc_os_1",
                    "page": 1,
                },
            },
            {
                "chunk_id": "os_notes_1_p2_c1",
                "text": "Virtual memory allows an operating system to compensate for physical memory shortages by temporarily transferring data from RAM to disk storage using paging.",
                "metadata": {
                    "document": "os_notes.pdf",
                    "document_id": "doc_os_1",
                    "page": 2,
                },
            },
            {
                "chunk_id": "network_notes_1_txt_c1",
                "text": "The Transmission Control Protocol ensures reliable, ordered, and error-checked delivery of a stream of packets between applications.",
                "metadata": {
                    "document": "network_notes.txt",
                    "document_id": "doc_net_1",
                    "page": None,
                },
            },
        ]

    def test_vector_store_ingestion_and_count(self):
        """Test adding chunks from document processor and verifying count."""
        added = self.vector_store.add_chunks(self.sample_chunks)
        self.assertEqual(added, 3)
        self.assertEqual(self.vector_store.count(), 3)

        docs = self.vector_store.get_indexed_documents()
        self.assertIn("os_notes.pdf", docs)
        self.assertIn("network_notes.txt", docs)

    def test_similarity_search_retrieval(self):
        """Test semantic search returns relevant chunks ordered by relevance."""
        self.vector_store.add_chunks(self.sample_chunks)

        results = self.vector_store.similarity_search("virtual memory and paging", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], RetrievedChunk)
        # The top result should be the virtual memory chunk
        self.assertEqual(results[0].chunk_id, "os_notes_1_p2_c1")
        self.assertEqual(results[0].page_number, 2)
        self.assertIn("os_notes.pdf (Page 2)", results[0].format_source_citation())

    def test_search_with_document_filter(self):
        """Test filtering retrieval by specific document_id."""
        self.vector_store.add_chunks(self.sample_chunks)

        results = self.vector_store.similarity_search(
            "computer hardware",
            top_k=5,
            document_id="doc_net_1",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata.get("document_id"), "doc_net_1")

    def test_pydantic_flashcard_validation(self):
        """Test Flashcard Pydantic schema validation and normalization."""
        card = Flashcard(
            question="What is virtual memory?",
            answer="A memory management technique that uses secondary storage as additional RAM.",
            difficulty="medium",
            topic="Operating Systems",
            source="os_notes.pdf (Page 2)",
        )
        self.assertEqual(card.difficulty, "Medium")
        self.assertEqual(card.question, "What is virtual memory?")
        self.assertEqual(card.source, "os_notes.pdf (Page 2)")

    def test_pydantic_flashcard_invalid_question(self):
        """Test that invalid/empty questions raise validation error."""
        with self.assertRaises(Exception):
            Flashcard(question="", answer="Some answer")

    def test_prompt_formatting(self):
        """Test prompt construction with chunks and source citations."""
        chunks = [
            RetrievedChunk(
                chunk_id="chunk_1",
                text="Paging divides memory into fixed-size frames.",
                metadata={"document": "test.pdf", "page": 4},
                score=0.95,
            )
        ]
        prompt = build_flashcard_prompt(
            topic="Paging",
            chunks=chunks,
            num_flashcards=3,
            difficulty="Hard",
        )
        self.assertIn("Paging divides memory", prompt)
        self.assertIn("test.pdf (Page 4)", prompt)
        self.assertIn("Target Difficulty Level: Hard", prompt)

    def test_json_extraction_from_markdown_fence(self):
        """Test extracting JSON wrapped in markdown code blocks."""
        raw_llm_text = """
Here are the generated flashcards:
```json
{
  "topic": "Operating Systems",
  "difficulty_target": "Medium",
  "flashcards": [
    {
      "question": "What is an operating system?",
      "answer": "Software that manages computer hardware resources.",
      "difficulty": "Easy",
      "topic": "Operating Systems",
      "source": "os_notes.pdf (Page 1)",
      "source_chunk_ids": ["c1"]
    }
  ]
}
```
Hope this helps!
"""
        parsed = extract_json_from_response(raw_llm_text)
        self.assertEqual(parsed["topic"], "Operating Systems")
        self.assertEqual(len(parsed["flashcards"]), 1)

    def test_deduplicate_flashcards(self):
        """Test deduplication removes duplicate or near-identical questions."""
        cards = [
            Flashcard(
                question="What is virtual memory?",
                answer="Technique using disk as RAM.",
                difficulty="Easy",
            ),
            Flashcard(
                question="What is virtual memory?",  # Exact duplicate
                answer="Different wording for same thing.",
                difficulty="Medium",
            ),
            Flashcard(
                question="Explain virtual memory?",  # Near duplicate after prefix removal
                answer="Yet another phrasing.",
                difficulty="Hard",
            ),
            Flashcard(
                question="What is paging in operating systems?",  # Distinct question
                answer="Dividing address spaces into blocks.",
                difficulty="Medium",
            ),
        ]
        deduped = deduplicate_flashcards(cards)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0].question, "What is virtual memory?")
        self.assertEqual(deduped[1].question, "What is paging in operating systems?")

    def test_relevance_evaluation(self):
        """Test content relevance verification between flashcards and source chunks."""
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                text="The kernel provides memory allocation, process scheduling, and device drivers.",
                metadata={"document": "os.pdf", "page": 1},
                score=0.9,
            )
        ]
        grounded_cards = [
            Flashcard(
                question="What functions does the kernel provide?",
                answer="The kernel provides memory allocation, process scheduling, and device drivers.",
                difficulty="Easy",
            )
        ]
        eval_result = evaluate_flashcard_relevance(grounded_cards, chunks)
        self.assertTrue(eval_result["is_relevant"])
        self.assertGreater(eval_result["average_relevance"], 0.4)

    def test_end_to_end_generation_pipeline(self):
        """Test complete generation pipeline using MockLLMClient and ChromaVectorStore."""
        self.vector_store.add_chunks(self.sample_chunks)

        req = GenerationRequest(
            topic="Virtual Memory",
            num_flashcards=2,
            difficulty="Medium",
            top_k=2,
        )

        mock_client = MockLLMClient()
        flashcard_set = generate_flashcards(
            request=req,
            vector_store=self.vector_store,
            llm_client=mock_client,
        )

        self.assertIsInstance(flashcard_set, FlashcardSet)
        self.assertEqual(flashcard_set.topic, "Virtual Memory")
        self.assertEqual(len(flashcard_set.flashcards), 2)

        for card in flashcard_set.flashcards:
            self.assertTrue(len(card.question) > 5)
            self.assertTrue(len(card.answer) > 5)
            self.assertIn(card.difficulty, ["Easy", "Medium", "Hard"])
            self.assertTrue(card.source)

        # Check metadata
        self.assertIn("relevance_report", flashcard_set.metadata)
        self.assertEqual(flashcard_set.metadata["retrieved_chunks_count"], 2)

    def test_empty_vector_store_raises_error(self):
        """Test that attempting to generate from empty store raises FlashcardGenerationError."""
        empty_store = ChromaVectorStore(
            collection_name="empty_test_collection",
            use_in_memory=True,
        )
        req = GenerationRequest(topic="Any Topic")
        with self.assertRaises(FlashcardGenerationError):
            generate_flashcards(req, empty_store)


if __name__ == "__main__":
    unittest.main(verbosity=2)
