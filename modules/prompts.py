"""
prompts.py

Prompt templates and system instructions for Member 2: RAG Pipeline and AI Flashcard Generation.
Ensures factual grounding, structured JSON output, Bloom's taxonomy difficulty calibration,
and accurate source citations.
"""

from typing import List
from modules.models import RetrievedChunk


SYSTEM_PROMPT = """You are an expert educational AI specialized in Retrieval-Augmented Generation (RAG) for active recall and flashcard creation.

Your task is to generate high-quality study flashcards based STRICTLY on the retrieved context chunks provided.

GUIDELINES:
1. FACTUAL GROUNDING: Rely ONLY on the provided context chunks. Do not hallucinate or use external knowledge not supported by the excerpts.
2. SOURCE CITATIONS: For every flashcard, attribute the exact source document name and page number from the context chunk header (e.g., "unit3.pdf (Page 4)").
3. RELEVANCE: Focus directly on the user's requested topic.
4. DIFFICULTY LEVELS:
   - Easy: Direct definitions, basic facts, key terms, or recall.
   - Medium: Conceptual understanding, processes, comparisons, or mechanism explanations.
   - Hard: In-depth problem solving, trade-off analysis, synthesis, or edge-case reasoning.
   If "Mixed" is requested, provide a balanced blend of Easy, Medium, and Hard cards.
5. QUALITY & CONCISENESS:
   - Questions should be clear, unambiguous, and test one concept at a time.
   - Answers should be thorough yet concise, informative, and complete.
6. NO DUPLICATES: Ensure every flashcard tests a distinct concept or question.
7. OUTPUT FORMAT: Respond ONLY with a valid JSON object matching the requested schema. Do not include markdown preamble, commentary, or conversational remarks.
"""


def format_context_chunks(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved context chunks into numbered sections with metadata."""
    if not chunks:
        return "No relevant context chunks found."

    formatted_sections = []
    for idx, chunk in enumerate(chunks, start=1):
        citation = chunk.format_source_citation()
        header = f"--- [Excerpt {idx}] Source: {citation} | Chunk ID: {chunk.chunk_id} ---"
        body = chunk.text.strip()
        formatted_sections.append(f"{header}\n{body}")

    return "\n\n".join(formatted_sections)


def build_flashcard_prompt(
    topic: str,
    chunks: List[RetrievedChunk],
    num_flashcards: int = 5,
    difficulty: str = "Medium",
) -> str:
    """
    Construct user prompt for LLM flashcard generation.

    Args:
        topic: The topic or concept requested.
        chunks: List of retrieved context chunks from ChromaDB.
        num_flashcards: Desired number of cards.
        difficulty: Desired difficulty ('Easy', 'Medium', 'Hard', 'Mixed').

    Returns:
        Formatted prompt string.
    """
    context_text = format_context_chunks(chunks)

    prompt = f"""Generate {num_flashcards} study flashcard(s) on the topic: "{topic}".

Target Difficulty Level: {difficulty}

STUDY MATERIAL CONTEXT EXCERPTS:
==================================================
{context_text}
==================================================

INSTRUCTIONS:
1. Formulate exactly {num_flashcards} distinct flashcards addressing "{topic}".
2. Base every question and answer strictly on the facts in the excerpts above.
3. For each flashcard, include:
   - "question": The question text.
   - "answer": The direct, accurate answer.
   - "difficulty": "Easy", "Medium", or "Hard".
   - "topic": "{topic}".
   - "source": The source citation from the excerpt header (e.g., "unit3.pdf (Page 4)").
   - "source_chunk_ids": The chunk ID(s) from the excerpt header supporting this card.

REQUIRED JSON OUTPUT FORMAT:
{{
  "topic": "{topic}",
  "difficulty_target": "{difficulty}",
  "flashcards": [
    {{
      "question": "Example question?",
      "answer": "Example answer derived directly from context.",
      "difficulty": "Medium",
      "topic": "{topic}",
      "source": "Document.pdf (Page 1)",
      "source_chunk_ids": ["chunk_id_1"]
    }}
  ]
}}

Provide ONLY the valid JSON object.
"""
    return prompt.strip()
