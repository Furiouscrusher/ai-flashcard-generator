"""
modules package

Contains modules for:
- Member 1: Document Processing and Data Preparation
- Member 2: RAG Pipeline and AI Flashcard Generation
- Member 3: Frontend Development and Integration (upcoming)
"""

from modules.document_processor import (
    DocumentProcessingError,
    clean_text,
    process_document,
    split_into_chunks,
)
from modules.models import (
    Flashcard,
    FlashcardSet,
    GenerationRequest,
    RetrievedChunk,
)
from modules.vector_store import (
    ChromaVectorStore,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
)
from modules.prompts import (
    SYSTEM_PROMPT,
    build_flashcard_prompt,
    format_context_chunks,
)
from modules.llm_client import (
    BaseLLMClient,
    GeminiLLMClient,
    GroqLLMClient,
    MockLLMClient,
    OpenAILLMClient,
    OpenRouterLLMClient,
    extract_json_from_response,
    get_llm_client,
)
from modules.flashcard_generator import (
    FlashcardGenerationError,
    deduplicate_flashcards,
    evaluate_flashcard_relevance,
    generate_flashcards,
)

__all__ = [
    # Member 1
    "process_document",
    "split_into_chunks",
    "clean_text",
    "DocumentProcessingError",
    # Member 2 Models
    "Flashcard",
    "FlashcardSet",
    "GenerationRequest",
    "RetrievedChunk",
    # Member 2 Vector DB & Embeddings
    "ChromaVectorStore",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    # Member 2 Prompts
    "SYSTEM_PROMPT",
    "build_flashcard_prompt",
    "format_context_chunks",
    # Member 2 LLM Integration
    "BaseLLMClient",
    "GeminiLLMClient",
    "GroqLLMClient",
    "OpenRouterLLMClient",
    "OpenAILLMClient",
    "MockLLMClient",
    "get_llm_client",
    "extract_json_from_response",
    # Member 2 Generator & Validation
    "generate_flashcards",
    "FlashcardGenerationError",
    "deduplicate_flashcards",
    "evaluate_flashcard_relevance",
]
