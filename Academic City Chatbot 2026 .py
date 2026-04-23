"""Academic City Chatbot 2026 - Retrieval-Augmented Generation (RAG) Chatbot

This script demonstrates a simple RAG system for local document retrieval
plus a text-generation layer. It supports OpenAI embeddings + completion if
configured, and falls back to a local retrieval-only mode when OpenAI is not
available.

Usage:
  python "Academic City Chatbot 2026 .py" --docs data --top-k 3

Requirements:
  - Python 3.8+
  - Optional: openai, numpy, scikit-learn, sentence-transformers

If OpenAI is configured with OPENAI_API_KEY, the chatbot will generate
answers using the OpenAI API. Otherwise the script will still retrieve
relevant context and return a retrieval summary.
"""

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Optional dependencies
try:
    import numpy as np
except ImportError:
    np = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    import openai
except ImportError:
    openai = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


@dataclass
class DocumentChunk:
    text: str
    source: str
    metadata: Dict[str, str] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class TextSplitter:
    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def split(text: str, chunk_size: int = 400, overlap: int = 100) -> List[str]:
        text = TextSplitter.normalize(text)
        words = text.split(" ")
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += chunk_size - overlap
        return chunks


class EmbeddingProvider:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None

    def get_embedding(self, text: str) -> List[float]:
        if openai is not None and os.getenv("OPENAI_API_KEY"):
            return self._openai_embedding(text)

        if self.model is not None:
            return self.model.encode(text).tolist()

        return self._fallback_embedding(text)

    def _openai_embedding(self, text: str) -> List[float]:
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response["data"][0]["embedding"]
        except Exception as exc:
            print(f"OpenAI embedding failed: {exc}")
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return [counts[k] for k in sorted(counts.keys())]


class VectorRetriever:
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.vector_matrix = None
        self.use_sklearn = TfidfVectorizer is not None and cosine_similarity is not None

        if self.use_sklearn:
            self._build_tfidf_index()
        elif np is not None:
            self._build_numpy_index()

    def _build_tfidf_index(self) -> None:
        self.vectorizer = TfidfVectorizer().fit([chunk.text for chunk in self.chunks])
        self.vector_matrix = self.vectorizer.transform([chunk.text for chunk in self.chunks])

    def _build_numpy_index(self) -> None:
        self.vector_matrix = np.array(
            [np.array(chunk.embedding or self._text_embedding(chunk.text), dtype=float)
             for chunk in self.chunks],
            dtype=float,
        )

    def _text_embedding(self, text: str) -> List[float]:
        return [len(text), sum(1 for c in text if c.isupper())]

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        if self.use_sklearn:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.vector_matrix)[0]
            ranked_indices = similarities.argsort()[::-1][:top_k]
            return [(self.chunks[i], float(similarities[i])) for i in ranked_indices if similarities[i] > 0]

        if np is not None and self.vector_matrix is not None:
            query_embed = self._simple_numpy_embedding(query)
            similarities = self._cosine_similarity(query_embed, self.vector_matrix)
            ranked_indices = np.argsort(similarities)[::-1][:top_k]
            return [(self.chunks[i], float(similarities[i])) for i in ranked_indices if similarities[i] > 0]

        # Fallback lexical search
        scored: List[Tuple[DocumentChunk, float]] = []
        query_tokens = set(re.findall(r"\w+", query.lower()))
        for chunk in self.chunks:
            chunk_tokens = set(re.findall(r"\w+", chunk.text.lower()))
            score = len(query_tokens & chunk_tokens)
            scored.append((chunk, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _simple_numpy_embedding(self, query: str) -> np.ndarray:
        words = re.findall(r"\w+", query.lower())
        vector = np.zeros(self.vector_matrix.shape[1], dtype=float)
        for word in words:
            idx = abs(hash(word)) % self.vector_matrix.shape[1]
            vector[idx] += 1
        return vector

    @staticmethod
    def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if np.linalg.norm(query) == 0 or np.any(np.linalg.norm(matrix, axis=1) == 0):
            return np.zeros(matrix.shape[0], dtype=float)
        dot = matrix.dot(query)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        return dot / norms


class RAGChatbot:
    def __init__(self, docs_path: str, top_k: int = 3, chunk_size: int = 400, overlap: int = 100):
        self.docs_path = docs_path
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding_provider = EmbeddingProvider()
        self.chunks = self._load_documents()
        self.retriever = VectorRetriever(self.chunks)

    def _load_documents(self) -> List[DocumentChunk]:
        path = Path(self.docs_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            raise FileNotFoundError(
                f"Documents folder created at {path}. Add .txt or .md files and rerun the script."
            )

        chunks: List[DocumentChunk] = []
        extensions = ["*.txt", "*.md"]
        file_paths: List[Path] = []
        for ext in extensions:
            file_paths.extend(sorted(path.glob(ext)))

        if not file_paths:
            sample_path = path / "sample_document.txt"
            sample_text = (
                "Academic City Chatbot Sample Document\n"
                "This sample document exists to start the chatbot.\n"
                "Ask the bot about the Academic City Chatbot or describe your own data to use it.\n"
            )
            sample_path.write_text(sample_text, encoding="utf-8")
            file_paths.append(sample_path)
            print(f"No documents found. Created sample document at {sample_path}.")

        for file_path in file_paths:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            for part in TextSplitter.split(text, self.chunk_size, self.overlap):
                chunks.append(DocumentChunk(
                    text=part,
                    source=str(file_path.name),
                    metadata={"source": str(file_path)}
                ))
        return chunks

    def answer(self, query: str) -> str:
        hits = self.retriever.search(query, top_k=self.top_k)
        if not hits:
            return "I could not find relevant documents. Please try a different question."

        context = "\n\n".join(
            f"Source: {chunk.source}\n{chunk.text}"
            for chunk, _score in hits
        )

        prompt = (
            "You are a helpful assistant with access to retrieved knowledge from local documents. "
            "Use the context to answer the user question factually. If the answer is not present, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )

        return self._generate_text(prompt)

    def _generate_text(self, prompt: str) -> str:
        if openai is not None and os.getenv("OPENAI_API_KEY"):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=300,
                )
                return response["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                print(f"OpenAI generation error: {exc}")

        # Fallback retrieval-only reply
        return self._retrieval_summary(prompt)

    def _retrieval_summary(self, prompt: str) -> str:
        lines = prompt.splitlines()
        question = [line for line in lines if line.startswith("Question:")]
        return (
            "Retrieval-only mode: OpenAI is not configured or unavailable. "
            "Here is the most relevant context I found:\n\n"
            + "\n\n".join(
                f"Source: {chunk.source}\n{chunk.text}" for chunk, _ in self.retriever.search(question[0].replace("Question:", "").strip(), top_k=self.top_k)
            )
        )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_docs = str(script_dir / "docs")

    parser = argparse.ArgumentParser(description="Academic City Chatbot 2026 - RAG system")
    parser.add_argument("--docs", type=str, default=default_docs, help="Folder containing text or markdown documents")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=400, help="Chunk size in words")
    parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap in words")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs_path = Path(args.docs).expanduser()
    if not docs_path.is_absolute():
        docs_path = Path(__file__).resolve().parent / docs_path
    try:
        bot = RAGChatbot(
            docs_path=str(docs_path.resolve()),
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    print("Academic City Chatbot 2026 — RAG system ready.")
    print("Type a question or 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        answer = bot.answer(question)
        print(f"\nBot: {answer}")


if __name__ == "__main__":
    main()
