# AI Flashcard Generator Using RAG

An AI-powered active recall flashcard generation application that converts uploaded educational documents (PDF and TXT) into structured study flashcards using Retrieval-Augmented Generation (RAG).

---

## Architecture & Work Distribution

### Member 1: Document Processing and Data Preparation (Completed)
- Document text extraction from PDF (PyMuPDF) and TXT.
- Text normalization, unicode cleaning, and newline standardizing.
- Word-safe text chunking with configurable chunk size and overlap.
- Metadata attribution: SHA-256 document IDs and PDF page numbers.

### Member 2: RAG Pipeline and AI Flashcard Generation (Completed)
- **Embeddings**: Dense text embeddings using Sentence Transformers (`all-MiniLM-L6-v2`) with in-memory caching.
- **Vector Database**: ChromaDB vector store with in-memory ephemeral and persistent storage support.
- **Semantic Search**: Top-k similarity retrieval with cosine distance scoring and metadata filtering (`document_id`).
- **LLM Multi-Provider Integration**:
  - **Groq API** (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) via `GROQ_API_KEY` (ultra-fast inference & generous free tier).
  - **OpenRouter API** (`meta-llama/llama-3.3-70b-instruct`, `deepseek/deepseek-chat`) via `OPENROUTER_API_KEY`.
  - **Google Gemini API** (`gemini-2.5-flash` / `gemini-1.5-flash`) via `google-genai`.
  - **OpenAI API** (`gpt-4o-mini` / `gpt-3.5-turbo`) via `openai`.
  - **Offline Mock LLM provider** (`MockLLMClient`) allowing full offline testing and grading without API keys.
- **Prompt Engineering**: Bloom's taxonomy calibrated difficulty prompts, strict context grounding, source citation enforcement, and structured JSON output.
- **Validation & Pydantic**: Robust schemas (`Flashcard`, `FlashcardSet`, `GenerationRequest`, `RetrievedChunk`).
- **Deduplication**: Intelligent normalized question deduplication.
- **Relevance Evaluation**: Automated vocabulary overlap and context grounding verification.

### Member 3: Frontend Development and Project Integration (Upcoming)
- Streamlit web interface for document uploads, topic selection, and flashcard presentation.
- Interactive flashcards with answer reveal, navigation, and JSON export.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Furiouscrusher/ai-flashcard-generator.git
   cd ai-flashcard-generator
   ```

2. **Create and activate virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **(Optional) Configure API Keys:**
   Copy `.env.example` to `.env` and set your API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   # or
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   *Note: If no API keys are provided, the system automatically runs with the built-in offline LLM provider.*

---

## Running Tests & Demonstrations

### 1. Run Complete Automated Unit Test Suite (18 Tests)
Runs both Member 1's document processor tests and Member 2's RAG pipeline tests:
```powershell
python -m unittest discover -s tests -v
```

### 2. Member 1 Document Processor Smoke Test
```powershell
python test_processor.py
```

### 3. Member 2 RAG Pipeline & Flashcard Smoke Test
Indexes `sample_docs/unit3.pdf`, performs semantic search on `"Cell Structure and Mitochondria"`, generates validated flashcards, runs relevance evaluation, and exports cards to `sample_flashcards.json`:
```powershell
python test_rag_pipeline.py
```

---

## Module Interfaces for Member 3 Integration

### 1. Ingesting Documents into ChromaDB
```python
from modules import process_document, ChromaVectorStore

# Process document (Member 1)
chunks = process_document(file_bytes=pdf_bytes, filename="notes.pdf")

# Index in ChromaDB (Member 2)
vector_store = ChromaVectorStore(persist_directory="./data/chroma_db")
vector_store.add_chunks(chunks)
```

### 2. Generating Flashcards from Topic
```python
from modules import GenerationRequest, generate_flashcards

request = GenerationRequest(
    topic="Mitochondria",
    num_flashcards=5,
    difficulty="Mixed",  # "Easy", "Medium", "Hard", or "Mixed"
    top_k=4,
)

flashcard_set = generate_flashcards(request=request, vector_store=vector_store)

# Access validated cards
for card in flashcard_set.flashcards:
    print(card.question)
    print(card.answer)
    print(card.difficulty)
    print(card.source)

# Export for UI download button
json_data = flashcard_set.to_dict_list()
```