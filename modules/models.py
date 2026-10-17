"""
models.py

Pydantic data models for Member 2: RAG Pipeline and AI Flashcard Generation.
Defines structured schemas for chunks, retrieval results, flashcards, and generation requests.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


DifficultyType = Literal["Easy", "Medium", "Hard"]
DifficultyFilterType = Literal["Easy", "Medium", "Hard", "Mixed"]


class RetrievedChunk(BaseModel):
    """Represents a text chunk retrieved from the vector database."""

    chunk_id: str = Field(description="Unique identifier for the chunk.")
    text: str = Field(description="Extracted and cleaned chunk text.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata including document name, page, and document_id.",
    )
    score: float = Field(
        default=0.0,
        description="Similarity or relevance score (higher is more relevant).",
    )

    @property
    def document_name(self) -> str:
        """Helper to get document name safely."""
        return str(self.metadata.get("document", "Unknown Document"))

    @property
    def page_number(self) -> Optional[int]:
        """Helper to get page number if present."""
        return self.metadata.get("page")

    def format_source_citation(self) -> str:
        """Format a human-readable citation string."""
        doc = self.document_name
        page = self.page_number
        if page is not None:
            return f"{doc} (Page {page})"
        return doc


class Flashcard(BaseModel):
    """Represents an individual AI-generated flashcard."""

    question: str = Field(
        ...,
        min_length=5,
        description="Clear, well-formed question testing understanding.",
    )
    answer: str = Field(
        ...,
        min_length=2,
        description="Accurate, concise answer grounded strictly in the source context.",
    )
    difficulty: DifficultyType = Field(
        default="Medium",
        description="Difficulty level of the question: Easy, Medium, or Hard.",
    )
    topic: Optional[str] = Field(
        default=None,
        description="Topic or concept to which the flashcard relates.",
    )
    source: Optional[str] = Field(
        default=None,
        description="Source document name and page number citation (e.g. 'unit3.pdf (Page 5)').",
    )
    source_chunk_ids: List[str] = Field(
        default_factory=list,
        description="List of chunk IDs that provided context for this flashcard.",
    )

    @field_validator("question", "answer", mode="before")
    @classmethod
    def clean_text_field(cls, value: Any) -> str:
        if not isinstance(value, str):
            value = str(value)
        return value.strip()

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, value: Any) -> str:
        if isinstance(value, str):
            val_lower = value.strip().lower()
            if "easy" in val_lower:
                return "Easy"
            if "hard" in val_lower:
                return "Hard"
            return "Medium"
        return "Medium"


class FlashcardSet(BaseModel):
    """Collection of generated flashcards with generation metadata."""

    topic: str = Field(description="The topic requested by the user.")
    difficulty_target: str = Field(
        default="Medium",
        description="Target difficulty level or 'Mixed'.",
    )
    flashcards: List[Flashcard] = Field(
        default_factory=list,
        description="List of validated flashcards.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operational metadata: timestamps, model name, chunks used, etc.",
    )

    @property
    def count(self) -> int:
        return len(self.flashcards)

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Export flashcards as list of dictionaries for frontend or JSON export."""
        return [card.model_dump() for card in self.flashcards]


class GenerationRequest(BaseModel):
    """Parameters for flashcard generation from a topic."""

    topic: str = Field(
        ...,
        min_length=1,
        description="Topic, keyword, or concept to generate flashcards for.",
    )
    num_flashcards: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of flashcards to generate.",
    )
    difficulty: DifficultyFilterType = Field(
        default="Medium",
        description="Target difficulty: Easy, Medium, Hard, or Mixed.",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional filter to restrict retrieval to a specific document.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant text chunks to retrieve for context.",
    )

    @field_validator("topic", mode="before")
    @classmethod
    def strip_topic(cls, value: Any) -> str:
        return str(value).strip()
