"""
flashcard_generator.py

Member 2: RAG Pipeline - AI Flashcard Generator & Output Validator.
Orchestrates semantic retrieval, prompt construction, LLM invocation, Pydantic validation,
deduplication, and content relevance evaluation.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

from modules.models import (
    Flashcard,
    FlashcardSet,
    GenerationRequest,
    RetrievedChunk,
)
from modules.prompts import SYSTEM_PROMPT, build_flashcard_prompt
from modules.llm_client import (
    BaseLLMClient,
    extract_json_from_response,
    get_llm_client,
)
from modules.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class FlashcardGenerationError(Exception):
    """Raised when flashcard generation fails."""


def normalize_question(question: str) -> str:
    """Normalize question text for deduplication comparison."""
    # Lowercase and remove punctuation
    cleaned = re.sub(r"[^\w\s]", "", question.lower()).strip()
    # Normalize common query prefixes
    cleaned = re.sub(r"^(what is|what are|explain|describe|define)\s+", "", cleaned)
    return " ".join(cleaned.split())


def deduplicate_flashcards(flashcards: List[Flashcard]) -> List[Flashcard]:
    """
    Remove duplicate or nearly identical flashcards based on normalized question text.
    Preserves the first occurrence.
    """
    seen: Set[str] = set()
    unique_cards: List[Flashcard] = []

    for card in flashcards:
        norm = normalize_question(card.question)
        if not norm or norm in seen:
            logger.info("Dropping duplicate flashcard: '%s'", card.question)
            continue
        seen.add(norm)
        unique_cards.append(card)

    return unique_cards


def evaluate_flashcard_relevance(
    flashcards: List[Flashcard],
    retrieved_chunks: List[RetrievedChunk],
) -> Dict[str, Any]:
    """
    Evaluate whether generated flashcards are relevant to and grounded in the retrieved content.
    Calculates vocabulary overlap and grounding score between card terms and source chunks.

    Returns:
        Dictionary with overall relevance score, grounded count, and per-card details.
    """
    if not flashcards or not retrieved_chunks:
        return {
            "is_relevant": False,
            "average_relevance": 0.0,
            "grounded_count": 0,
            "total_cards": len(flashcards),
            "details": [],
        }

    # Aggregate all source words (ignoring short stopwords)
    source_words = set()
    for chunk in retrieved_chunks:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", chunk.text.lower())
        source_words.update(words)

    card_scores = []
    grounded_count = 0

    for card in flashcards:
        card_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", f"{card.question} {card.answer}".lower()))
        if not card_words:
            card_scores.append(0.0)
            continue

        overlap = card_words.intersection(source_words)
        score = len(overlap) / len(card_words)
        card_scores.append(round(score, 3))

        # A card is considered grounded if at least 40% of its content terms appear in the retrieved excerpts
        if score >= 0.40:
            grounded_count += 1

    avg_score = round(sum(card_scores) / len(card_scores), 3) if card_scores else 0.0
    is_relevant = (grounded_count / len(flashcards)) >= 0.60 if flashcards else False

    return {
        "is_relevant": is_relevant,
        "average_relevance": avg_score,
        "grounded_count": grounded_count,
        "total_cards": len(flashcards),
        "scores": card_scores,
    }


def generate_flashcards(
    request: GenerationRequest,
    vector_store: ChromaVectorStore,
    llm_client: Optional[BaseLLMClient] = None,
) -> FlashcardSet:
    """
    Execute the complete RAG flashcard generation pipeline:
    1. Semantic search for relevant chunks based on user topic.
    2. Prompt construction with context grounding and source citations.
    3. Invocation of LLM (Gemini, OpenAI, or Mock fallback).
    4. Response parsing and Pydantic validation.
    5. Deduplication and source attribution.
    6. Content relevance testing.

    Args:
        request: GenerationRequest specifying topic, count, difficulty, and filters.
        vector_store: Initialized ChromaVectorStore containing document chunks.
        llm_client: Optional BaseLLMClient instance; if None, uses default auto-detected provider.

    Returns:
        A validated FlashcardSet object.

    Raises:
        FlashcardGenerationError: If retrieval or generation fails.
    """
    if vector_store.count() == 0:
        raise FlashcardGenerationError(
            "Vector database is empty. Upload and process study documents first."
        )

    # 1. Semantic Search
    logger.info("Performing semantic search for topic: '%s'", request.topic)
    chunks = vector_store.similarity_search(
        query=request.topic,
        top_k=request.top_k,
        document_id=request.document_id,
    )

    if not chunks:
        raise FlashcardGenerationError(
            f"No relevant study material found for topic: '{request.topic}'."
        )

    # 2. Build Prompt
    prompt = build_flashcard_prompt(
        topic=request.topic,
        chunks=chunks,
        num_flashcards=request.num_flashcards,
        difficulty=request.difficulty,
    )

    # 3. Invoke LLM Client
    client = llm_client or get_llm_client()
    logger.info("Calling LLM client (%s) to generate flashcards...", client.__class__.__name__)

    try:
        raw_response = client.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
    except Exception as exc:
        raise FlashcardGenerationError(f"LLM generation failed: {exc}") from exc

    # 4. Parse JSON & Validate with Pydantic
    try:
        parsed_data = extract_json_from_response(raw_response)
    except ValueError as exc:
        raise FlashcardGenerationError(
            f"Invalid response format received from LLM: {exc}"
        ) from exc

    raw_cards = parsed_data.get("flashcards", [])
    if not isinstance(raw_cards, list) or not raw_cards:
        raise FlashcardGenerationError("LLM response did not contain a valid list of flashcards.")

    validated_cards: List[Flashcard] = []
    default_source = chunks[0].format_source_citation()
    default_chunk_ids = [c.chunk_id for c in chunks[:2]]

    for item in raw_cards:
        if not isinstance(item, dict):
            continue
        try:
            # Set default source and chunk ids if missing
            if not item.get("source"):
                item["source"] = default_source
            if not item.get("source_chunk_ids"):
                item["source_chunk_ids"] = default_chunk_ids
            if not item.get("topic"):
                item["topic"] = request.topic

            card = Flashcard(**item)
            validated_cards.append(card)
        except Exception as card_err:
            logger.warning("Skipping invalid flashcard: %s. Error: %s", item, card_err)

    if not validated_cards:
        raise FlashcardGenerationError(
            "None of the generated flashcards satisfied validation rules."
        )

    # 5. Deduplication
    unique_cards = deduplicate_flashcards(validated_cards)

    # Trim to requested number if extra cards were generated
    final_cards = unique_cards[: request.num_flashcards]

    # 6. Evaluate Relevance
    relevance_report = evaluate_flashcard_relevance(final_cards, chunks)

    # Compile FlashcardSet with full metadata
    result_set = FlashcardSet(
        topic=request.topic,
        difficulty_target=request.difficulty,
        flashcards=final_cards,
        metadata={
            "retrieved_chunks_count": len(chunks),
            "chunks_used": [c.chunk_id for c in chunks],
            "llm_provider": client.__class__.__name__,
            "relevance_report": relevance_report,
            "source_documents": sorted(list({c.document_name for c in chunks})),
        },
    )

    return result_set
