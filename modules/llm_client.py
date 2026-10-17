"""
llm_client.py

LLM integration client supporting Google Gemini, OpenAI, and a Mock provider.
Handles API authentication, prompt execution, error recovery, and JSON extraction.
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

logger = logging.getLogger(__name__)


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """
    Extract and parse a JSON dictionary from an LLM response string.
    Handles raw JSON, markdown code fences (```json ... ```), and trailing text.
    """
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Look for markdown code fence ```json ... ``` or ``` ... ```
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(fence_pattern, text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Look for the first balanced curly-bracket block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidate = text[brace_start : brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse valid JSON from LLM response:\n{text[:500]}")


class BaseLLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate raw text response from the model."""
        pass


class GeminiLLMClient(BaseLLMClient):
    """Client for Google Gemini models via google-genai or google-generativeai."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        self.model_name = model
        self._init_client()

    def _init_client(self):
        # Attempt google-genai first
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self._sdk_type = "google-genai"
            return
        except ImportError:
            pass

        # Attempt legacy google-generativeai
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=self.api_key)
            self.client = genai_legacy.GenerativeModel(self.model_name)
            self._sdk_type = "legacy"
            return
        except ImportError as exc:
            raise ImportError(
                "Neither google-genai nor google-generativeai is installed."
            ) from exc

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if self._sdk_type == "google-genai":
            config = {}
            if system_prompt:
                config["system_instruction"] = system_prompt
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config if config else None,
            )
            return response.text or ""
        else:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            response = self.client.generate_content(full_prompt)
            return response.text or ""


class OpenAILLMClient(BaseLLMClient):
    """Client for OpenAI models (GPT-4o, GPT-4o-mini)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY.")
        self.model_name = model
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError as exc:
            raise ImportError("openai package is not installed.") from exc

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "{}"


class MockLLMClient(BaseLLMClient):
    """
    Offline/Fallback LLM client for testing, development, and grading without paid API keys.
    Extracts key conceptual statements and definitions from provided context chunks
    to synthesize grounded, structured flashcards.
    """

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Extract topic from prompt
        topic_match = re.search(r'topic: "([^"]+)"', prompt, re.IGNORECASE)
        topic = topic_match.group(1) if topic_match else "Study Material"

        num_cards_match = re.search(r"Generate (\d+) study flashcard", prompt, re.IGNORECASE)
        num_cards = int(num_cards_match.group(1)) if num_cards_match else 3

        diff_match = re.search(r"Target Difficulty Level:\s*([A-Za-z]+)", prompt, re.IGNORECASE)
        target_diff = diff_match.group(1) if diff_match else "Medium"

        # Parse context excerpts
        excerpts = re.findall(
            r"--- \[Excerpt \d+\] Source: ([^|]+) \| Chunk ID: ([^\s]+) ---\s*([\s\S]*?)(?=(?:--- \[Excerpt|\Z))",
            prompt,
        )

        cards = []
        difficulties = ["Easy", "Medium", "Hard"]

        if not excerpts:
            # Fallback if no excerpts matched
            for i in range(num_cards):
                cards.append({
                    "question": f"What is a primary concept of {topic} (Part {i+1})?",
                    "answer": f"{topic} involves systematic study and application of core domain principles.",
                    "difficulty": target_diff if target_diff != "Mixed" else difficulties[i % 3],
                    "topic": topic,
                    "source": "Study Material",
                    "source_chunk_ids": [],
                })
        else:
            card_idx = 0
            while len(cards) < num_cards and card_idx < 50:
                excerpt = excerpts[card_idx % len(excerpts)]
                source_cite = excerpt[0].strip()
                chunk_id = excerpt[1].strip()
                chunk_text = excerpt[2].strip()

                # Find sentences that look like definitions or explanations
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text) if len(s.strip()) > 30]

                if sentences:
                    sentence = sentences[card_idx % len(sentences)]
                    # Create question from sentence
                    first_words = sentence.split()[:4]
                    subject = " ".join(first_words)
                    question = f"According to the source, what does the text explain about {subject}?"
                    answer = sentence
                else:
                    question = f"What key information does the source discuss regarding {topic}?"
                    answer = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text

                diff = target_diff if target_diff != "Mixed" else difficulties[len(cards) % 3]

                cards.append({
                    "question": question,
                    "answer": answer,
                    "difficulty": diff,
                    "topic": topic,
                    "source": source_cite,
                    "source_chunk_ids": [chunk_id],
                })
                card_idx += 1

        result_dict = {
            "topic": topic,
            "difficulty_target": target_diff,
            "flashcards": cards[:num_cards],
        }
        return json.dumps(result_dict, indent=2)


def get_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLMClient:
    """
    Factory to return an appropriate LLM client.

    If provider is not explicitly set, auto-detects from available environment variables:
    1. GEMINI_API_KEY / GOOGLE_API_KEY -> GeminiLLMClient
    2. OPENAI_API_KEY -> OpenAILLMClient
    3. None -> MockLLMClient (logs informational notice)
    """
    provider_str = (provider or "").strip().lower()

    if provider_str in ("gemini", "google"):
        return GeminiLLMClient(api_key=api_key, model=model or "gemini-2.5-flash")

    if provider_str == "openai":
        return OpenAILLMClient(api_key=api_key, model=model or "gpt-4o-mini")

    if provider_str == "mock":
        return MockLLMClient()

    # Auto-detection
    gemini_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            return GeminiLLMClient(api_key=gemini_key, model=model or "gemini-2.5-flash")
        except Exception as e:
            logger.warning("Failed to initialize Gemini client: %s. Falling back to Mock.", e)

    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            return OpenAILLMClient(api_key=openai_key, model=model or "gpt-4o-mini")
        except Exception as e:
            logger.warning("Failed to initialize OpenAI client: %s. Falling back to Mock.", e)

    logger.info("No active API keys found for Gemini or OpenAI. Using MockLLMClient.")
    return MockLLMClient()
