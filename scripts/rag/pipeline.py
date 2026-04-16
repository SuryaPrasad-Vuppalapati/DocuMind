from typing import Dict, List, Tuple, Optional
import uuid
import json

from .answer_generator import LegacyAnswerGenerator as AnswerGenerator
from .chunk_store import ChunkStore
from .chunker import TextChunker
from .client import OpenAIClient
from .corpus_loader import CorpusLoader
from .embedder import Embedder
from .indexer import FaissIndexer
from .retriever import Retriever
from rank_bm25 import BM25Okapi


class RAGPipeline:
    def __init__(
        self,
        corpus_path: str = "data/corpus.json",
        index_path: str = "data/my_index.faiss",
        chunks_path: str = "data/chunks.json",
        chunk_size: int = 1500,
        overlap: int = 200,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
    ) -> None:
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.loader = CorpusLoader(corpus_path)
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.client = OpenAIClient().client
        self.embedder = Embedder(self.client, embedding_model=embedding_model)
        self.indexer = FaissIndexer()
        self.chunk_store = ChunkStore()
        self.retriever = Retriever(self.embedder)

        # AnswerGenerator now contains: session_memory, query_expander, intent_classifier
        self.answer_generator = AnswerGenerator(self.client, model=chat_model)

        self.loaded_index = None
        self.loaded_chunks = None
        self.loaded_bm25 = None
        self.current_session_id = None

    def build(self) -> int:
        corpus = self.loader.load()
        chunks = self.chunker.chunk_corpus(corpus)
        vectors = self.embedder.embed_chunks(chunks)
        index = self.indexer.build(vectors)
        self.indexer.save(index, self.index_path)
        self.chunk_store.save(chunks, self.chunks_path)

        self.loaded_index = index
        self.loaded_chunks = chunks
        self.loaded_bm25 = None  # force rebuild
        return len(chunks)

    def load_if_needed(self):
        if self.loaded_index is None:
            self.loaded_index = self.indexer.load(self.index_path)
        if self.loaded_chunks is None:
            self.loaded_chunks = self.chunk_store.load(self.chunks_path)
        if self.loaded_bm25 is None and self.loaded_chunks is not None:
            tokenized_corpus = [chunk['text'].lower().split() for chunk in self.loaded_chunks]
            self.loaded_bm25 = BM25Okapi(tokenized_corpus)

    # ── Session management ──────────────────────────────────
    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        self.answer_generator.session_memory.ensure_session(session_id)
        self.current_session_id = session_id
        return session_id

    def set_session(self, session_id: str) -> None:
        self.current_session_id = session_id
        self.answer_generator.session_memory.ensure_session(session_id)

    # ── Convenience: full ask flow with session + expansion ─
    def ask(
        self,
        query: str,
        session_id: Optional[str] = None,
        method: str = "vector",
        k: int = 5,
    ) -> Tuple[str, List[Dict]]:
        """
        End-to-end ask:
        1. Classify intent (meta / injection / document)
        2. For document queries: expand query → retrieve with all variants → merge
        3. Generate structured JSON answer
        4. Update session memory
        Returns (raw_json_answer, retrieved_chunks).
        """
        if session_id:
            self.set_session(session_id)
        self.load_if_needed()

        ag = self.answer_generator
        sid = self.current_session_id

        intent = ag.classify_intent(query, sid)

        results: List[Dict] = []
        if intent == "document":
            variants = ag.expand_query(query, sid)
            seen, merged = set(), []
            for variant in variants:
                hits = self.retriever.retrieve(
                    query=variant,
                    index=self.loaded_index,
                    chunks=self.loaded_chunks,
                    bm25=self.loaded_bm25,
                    method=method,
                    k=k,
                )
                for chunk in hits:
                    key = (chunk["source"], chunk["page"], chunk["text"][:80])
                    if key not in seen:
                        seen.add(key)
                        merged.append(chunk)
            merged.sort(key=lambda c: c.get("score", 0), reverse=True)
            results = merged[: k * 2]

        answer = ag.generate(
            query=query,
            retrieved_chunks=results,
            session_id=sid,
            intent=intent,
        )

        return answer, results

    def run_full(self, query: str, method: str = "vector", k: int = 5) -> Tuple[str, List[Dict], int]:
        total_chunks = self.build()
        answer, results = self.ask(query=query, method=method, k=k)
        return answer, results, total_chunks