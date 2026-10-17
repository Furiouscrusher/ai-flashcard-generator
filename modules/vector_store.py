"""
vector_store.py

Member 2: RAG Pipeline - Embeddings & ChromaDB Vector Storage.
Handles text embedding generation via Sentence Transformers, vector storage in ChromaDB,
and semantic similarity search with metadata filtering.
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from modules.models import RetrievedChunk

logger = logging.getLogger(__name__)

# Default model recommended for lightweight, high-performance dense embeddings
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "study_flashcards"


# Cache shared model instances to prevent reloading weights repeatedly
_SHARED_EMBEDDERS: Dict[str, Any] = {}


class ChromaVectorStore:
    """
    Manages vector storage and semantic retrieval using ChromaDB and Sentence Transformers.
    Supports in-memory or persistent storage.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        use_in_memory: bool = False,
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Path to persist directory, or None for default './data/chroma_db'.
            embedding_model_name: Sentence Transformers model name.
            use_in_memory: If True, uses ephemeral in-memory storage (ideal for tests).
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.use_in_memory = use_in_memory

        if persist_directory is None and not use_in_memory:
            persist_directory = "./data/chroma_db"
        self.persist_directory = persist_directory

        self._init_embedder()
        self._init_chroma()

    def _init_embedder(self):
        """Initialize SentenceTransformer model or fallback embedding function."""
        try:
            if self.embedding_model_name not in _SHARED_EMBEDDERS:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer model: %s", self.embedding_model_name)
                _SHARED_EMBEDDERS[self.embedding_model_name] = SentenceTransformer(self.embedding_model_name)
            self.model = _SHARED_EMBEDDERS[self.embedding_model_name]
            self._has_sentence_transformer = True
        except Exception as e:
            logger.warning(
                "Could not load SentenceTransformer (%s). Chroma default embedder will be used.",
                e,
            )
            self.model = None
            self._has_sentence_transformer = False

    def _init_chroma(self):
        """Initialize ChromaDB client and collection."""
        import chromadb

        if self.use_in_memory or self.persist_directory is None:
            self.client = chromadb.Client()
        else:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of texts using Sentence Transformers."""
        if not texts:
            return []
        if self._has_sentence_transformer and self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.tolist()
        return []

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata dictionary for ChromaDB compatibility.
        Chroma requires metadata values to be str, int, float, or bool.
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                # Store None as empty string or 'None'
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add or update document chunks in ChromaDB.

        Args:
            chunks: List of chunk dictionaries formatted by Member 1's process_document:
                    {
                        "chunk_id": str,
                        "text": str,
                        "metadata": {
                            "document": str,
                            "document_id": str,
                            "page": int or None
                        }
                    }

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "").strip()
            if not chunk_id or not text:
                continue

            raw_meta = chunk.get("metadata", {})
            meta = self._sanitize_metadata(raw_meta)
            # Ensure chunk_id is recorded in metadata
            meta["chunk_id"] = chunk_id

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(meta)

        if not ids:
            return 0

        # Compute embeddings via Sentence Transformers if available
        embeddings = self.compute_embeddings(documents) if self._has_sentence_transformer else None

        if embeddings:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        logger.info("Successfully added %d chunks to ChromaDB collection '%s'", len(ids), self.collection_name)
        return len(ids)

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Perform semantic search in ChromaDB based on a topic or query.

        Args:
            query: User's selected topic or query string.
            top_k: Number of most relevant chunks to retrieve.
            document_id: Optional document_id filter to scope retrieval.

        Returns:
            List of RetrievedChunk instances ordered by descending relevance.
        """
        query_text = (query or "").strip()
        if not query_text or self.collection.count() == 0:
            return []

        # Build optional where clause
        where_filter = None
        if document_id:
            where_filter = {"document_id": document_id}

        limit = min(top_k, self.collection.count())
        if limit <= 0:
            return []

        if self._has_sentence_transformer and self.model is not None:
            query_embeddings = self.compute_embeddings([query_text])
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=limit,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=limit,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

        retrieved_chunks = []
        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for idx in range(len(ids_list)):
            chunk_id = ids_list[idx]
            text = docs_list[idx]
            meta = metas_list[idx] if idx < len(metas_list) else {}
            distance = dists_list[idx] if idx < len(dists_list) else 0.0

            # Convert cosine distance to approximate similarity score in [0, 1]
            similarity_score = max(0.0, 1.0 - (distance / 2.0))

            # Restore None for empty page metadata
            cleaned_meta = dict(meta)
            if cleaned_meta.get("page") == "" or cleaned_meta.get("page") == "None":
                cleaned_meta["page"] = None
            elif isinstance(cleaned_meta.get("page"), str) and cleaned_meta.get("page").isdigit():
                cleaned_meta["page"] = int(cleaned_meta.get("page"))

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=cleaned_meta,
                    score=round(similarity_score, 4),
                )
            )

        return retrieved_chunks

    def count(self) -> int:
        """Return total number of chunks stored in collection."""
        return self.collection.count()

    def get_indexed_documents(self) -> List[str]:
        """Return list of distinct document names currently indexed."""
        count = self.collection.count()
        if count == 0:
            return []
        data = self.collection.get(include=["metadatas"])
        docs = set()
        for meta in data.get("metadatas", []):
            if meta and "document" in meta:
                docs.add(meta["document"])
        return sorted(list(docs))

    def clear(self) -> None:
        """Clear all records from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._init_chroma()
