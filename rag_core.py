# Name: Sumnima Wuni | Index Number: 10022300194
import csv
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pypdf
from election_helpers import inject_election_aggregate
from openai import OpenAI

try:
    import config as app_config
except ImportError:
    app_config = None
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import faiss
except ImportError:
    faiss = None


def _clean_api_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = value.strip().strip('"').strip("'")
    if not s:
        return None
    lower = s.lower()
    placeholders = {
        "your_key_here",
        "your_groq_key_here",
        "replace_me",
        "xxx",
        "sk-your-key-here",
    }
    if lower in placeholders:
        return None
    return s


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    strategy: str
    metadata: Dict[str, str] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def split_fixed(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    words = normalize_text(text).split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def split_structure_aware(text: str, max_words: int = 500, overlap: int = 80) -> List[str]:
    cleaned = normalize_text(text)
    sections = re.split(r"(?=(SECTION\s+\d+|Appendix\s+\d+[A-Z]?))", cleaned, flags=re.IGNORECASE)
    merged: List[str] = []
    pending = ""
    for part in sections:
        if not part:
            continue
        if len((pending + " " + part).split()) < max_words:
            pending = (pending + " " + part).strip()
        else:
            if pending:
                merged.append(pending)
            pending = part
    if pending:
        merged.append(pending)

    chunked: List[str] = []
    for section_text in merged:
        piece = split_fixed(section_text, chunk_size=max_words, overlap=overlap)
        chunked.extend(piece)
    return chunked


class DocumentIngestor:
    def __init__(self, docs_path: str):
        self.docs_path = Path(docs_path)
        self.docs_path.mkdir(parents=True, exist_ok=True)

    def ingest(self, strategy: str = "fixed") -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for path in sorted(self.docs_path.glob("*")):
            suffix = path.suffix.lower()
            if suffix in {".txt", ".md"}:
                chunks.extend(self._ingest_text(path, strategy))
            elif suffix == ".pdf":
                chunks.extend(self._ingest_pdf(path, strategy))
            elif suffix == ".csv":
                chunks.extend(self._ingest_csv(path, strategy))
        return chunks

    def _chunk_text(self, text: str, strategy: str) -> List[str]:
        if strategy == "structure":
            return split_structure_aware(text)
        return split_fixed(text)

    def _ingest_text(self, path: Path, strategy: str) -> List[DocumentChunk]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        out: List[DocumentChunk] = []
        for idx, part in enumerate(self._chunk_text(text, strategy)):
            out.append(
                DocumentChunk(
                    chunk_id=f"{path.stem}-{strategy}-{idx}",
                    text=part,
                    source=path.name,
                    strategy=strategy,
                    metadata={"doc_type": "text", "token_count": str(estimate_tokens(part))},
                )
            )
        return out

    def _ingest_pdf(self, path: Path, strategy: str) -> List[DocumentChunk]:
        out: List[DocumentChunk] = []
        reader = pypdf.PdfReader(str(path))
        for page_no, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for idx, part in enumerate(self._chunk_text(page_text, strategy)):
                out.append(
                    DocumentChunk(
                        chunk_id=f"{path.stem}-p{page_no}-{strategy}-{idx}",
                        text=part,
                        source=path.name,
                        strategy=strategy,
                        metadata={
                            "doc_type": "pdf",
                            "page": str(page_no),
                            "token_count": str(estimate_tokens(part)),
                        },
                    )
                )
        return out

    def _ingest_csv(self, path: Path, strategy: str) -> List[DocumentChunk]:
        out: List[DocumentChunk] = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return out
        for i, row in enumerate(rows):
            row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            parts = self._chunk_text(row_text, strategy)
            for idx, part in enumerate(parts):
                out.append(
                    DocumentChunk(
                        chunk_id=f"{path.stem}-row{i}-{strategy}-{idx}",
                        text=part,
                        source=path.name,
                        strategy=strategy,
                        metadata={"doc_type": "csv", "row": str(i), "token_count": str(estimate_tokens(part))},
                    )
                )
        return out


class EmbeddingPipeline:
    """Local hashing embeddings only (no external API)."""

    def __init__(self, model_name: str = "hashing-768"):
        self.model_name = model_name
        self.local_vectorizer = HashingVectorizer(
            n_features=768,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        matrix = self.local_vectorizer.transform(texts).toarray()
        return np.array(matrix, dtype="float32")


class HybridRetriever:
    def __init__(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings
        self.use_faiss = faiss is not None
        self.index = None
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, query_emb: np.ndarray, top_k: int = 4) -> List[Dict[str, object]]:
        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(query_emb, top_k * 3)
            vector_scores = {int(i): float(s) for i, s in zip(indices[0], scores[0]) if i >= 0}
        else:
            sim = cosine_similarity(query_emb, self.embeddings)[0]
            rank = np.argsort(sim)[::-1][: top_k * 3]
            vector_scores = {int(i): float(sim[i]) for i in rank}

        q_tfidf = self.vectorizer.transform([query])
        keyword_scores_arr = cosine_similarity(q_tfidf, self.tfidf_matrix)[0]
        keywords = re.findall(r"\w+", query.lower())
        is_acronym_query = any(t.isupper() for t in query.split())
        domain_boost_terms = {"abfa", "petroleum", "debt", "allocation", "gdp"}

        combined: List[Tuple[int, float, float, float]] = []
        for idx, v_score in vector_scores.items():
            kw_score = float(keyword_scores_arr[idx])
            boost = 0.0
            lower_text = self.chunks[idx].text.lower()
            if any(term in lower_text for term in domain_boost_terms.intersection(set(keywords))):
                boost += 0.08
            if is_acronym_query and any(tok in self.chunks[idx].text for tok in query.split() if tok.isupper()):
                boost += 0.1
            score = 0.75 * v_score + 0.25 * kw_score + boost
            combined.append((idx, score, v_score, kw_score))

        combined.sort(key=lambda x: x[1], reverse=True)
        results: List[Dict[str, object]] = []
        for idx, score, v_score, kw_score in combined[:top_k]:
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "text": chunk.text,
                    "combined_score": score,
                    "vector_score": v_score,
                    "keyword_score": kw_score,
                    "metadata": chunk.metadata,
                }
            )
        return results


class PromptBuilder:
    @staticmethod
    def build(query: str, retrieved: List[Dict[str, object]], max_context_chars: int = 6000) -> str:
        context_blocks: List[str] = []
        total = 0
        for i, item in enumerate(retrieved, start=1):
            page = item["metadata"].get("page", "n/a")
            block = (
                f"[{i}] source={item['source']} page={page} chunk_id={item['chunk_id']} "
                f"score={item['combined_score']:.4f}\n{item['text']}\n"
            )
            if total + len(block) > max_context_chars:
                break
            context_blocks.append(block)
            total += len(block)

        context = "\n".join(context_blocks)
        return (
            "You are an Academic City RAG assistant. Answer ONLY from the retrieved context below.\n"
            "Rules:\n"
            "- If a block starts with AGGREGATED NATIONAL TOTALS, use it as the authoritative answer for national vote totals / who had the most votes that year.\n"
            "- Be clear and concise: short sentences or bullet points; aim under 120 words unless the question needs a small table.\n"
            "- No preamble (no 'Based on the context...'). Start with the direct answer.\n"
            "- Only if nothing in the context answers the question, reply exactly: I could not find this in provided sources.\n"
            "- Cite each factual claim with [source filename, page or chunk id].\n\n"
            f"Question:\n{query}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer:"
        )


def _token_set(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta = _token_set(a)
    tb = _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def _groq_key_resolved() -> Optional[str]:
    env_key = _clean_api_key(os.getenv("GROQ_API_KEY"))
    if env_key:
        return env_key
    if app_config is not None:
        return _clean_api_key(getattr(app_config, "GROQ_API_KEY", None))
    return None


class Generator:
    """Groq-only chat completions."""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self._groq_key = _groq_key_resolved()
        self.provider = "none"
        self.model = os.getenv("GROQ_CHAT_MODEL", model)
        self.client = None

        if self._groq_key:
            self.provider = "groq"
            self.client = OpenAI(api_key=self._groq_key, base_url="https://api.groq.com/openai/v1")

    def _chat(self, messages: list, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.05,
            max_tokens=max_tokens,
            messages=messages,
        )
        return resp.choices[0].message.content or ""

    def complete(self, prompt: str) -> str:
        if self.client is None:
            return "No LLM key set. Retrieval completed; review retrieved chunks as grounded answer evidence."
        try:
            return self._chat([{"role": "user", "content": prompt}], 320)
        except Exception as exc:
            return f"Model call failed: {exc}\nSet GROQ_API_KEY or edit config.py GROQ_API_KEY."

    def pure_llm(self, query: str) -> str:
        if self.client is None:
            return "No Groq API key set for model-only baseline."
        try:
            return self._chat([{"role": "user", "content": query}], 260)
        except Exception as exc:
            return f"Pure LLM call failed: {exc}"


class RAGChatbot:
    def __init__(self, docs_path: str = "docs", strategy: str = "fixed", top_k: int = 4):
        self.docs_path = docs_path
        self.strategy = strategy
        self.top_k = top_k
        self.log_path = Path("logs/runs.jsonl")
        self.feedback_path = Path("logs/feedback.jsonl")
        self.experiment_path = Path("logs/experiments.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.ingestor = DocumentIngestor(docs_path=docs_path)
        self.chunks = self.ingestor.ingest(strategy=strategy)
        if not self.chunks:
            raise ValueError("No documents found in docs/. Add CSV/PDF/TXT/MD files first.")
        self.embedder = EmbeddingPipeline()
        self.embeddings = self.embedder.encode([c.text for c in self.chunks])
        self.retriever = HybridRetriever(self.chunks, self.embeddings)
        self.generator = Generator()

    @staticmethod
    def answer_from_retrieval_only(retrieved: List[Dict[str, object]]) -> str:
        if not retrieved:
            return "No matching excerpts found in docs/."
        lines = [
            "**From your data (no API key):** top matching excerpts. "
            "Add a Groq or OpenAI key in the app to get a short written answer.\n"
        ]
        for i, item in enumerate(retrieved[:3], 1):
            page = item["metadata"].get("page", "—")
            snip = item["text"][:420].replace("\n", " ").strip()
            lines.append(f"\n**{i}.** `{item['source']}` · p.{page}  \n{snip}…")
        return "\n".join(lines)

    def query(self, user_query: str, include_pure_baseline: bool = False) -> Dict[str, object]:
        """include_pure_baseline=True runs a second LLM call (slower). Default False for faster UI."""
        t0 = time.perf_counter()
        cleaned_query = normalize_text(user_query)
        query_emb = self.embedder.encode([cleaned_query])
        t1 = time.perf_counter()
        retrieved = self.retriever.search(cleaned_query, query_emb, top_k=self.top_k)
        retrieved = inject_election_aggregate(cleaned_query, retrieved, str(self.docs_path))
        retrieved = retrieved[: self.top_k]
        t2 = time.perf_counter()
        prompt = PromptBuilder.build(cleaned_query, retrieved)
        t3 = time.perf_counter()
        answer = self.generator.complete(prompt)
        if isinstance(answer, str) and answer.startswith("No LLM key set"):
            answer = self.answer_from_retrieval_only(retrieved)
        t4 = time.perf_counter()
        pure_llm_answer = ""
        t5 = t4
        if include_pure_baseline and self.generator.client is not None:
            pure_llm_answer = self.generator.pure_llm(cleaned_query)
            t5 = time.perf_counter()
        elif include_pure_baseline:
            pure_llm_answer = "No Groq client for model-only baseline."

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "strategy": self.strategy,
            "query": cleaned_query,
            "retrieved": retrieved,
            "prompt": prompt,
            "answer": answer,
            "pure_llm_answer": pure_llm_answer,
            "latency_ms": {
                "embed": round((t1 - t0) * 1000, 2),
                "retrieve": round((t2 - t1) * 1000, 2),
                "prompt": round((t3 - t2) * 1000, 2),
                "generate_rag": round((t4 - t3) * 1000, 2),
                "generate_pure_baseline_ms": round((t5 - t4) * 1000, 2) if include_pure_baseline else 0.0,
                "total": round((t5 - t0) * 1000, 2),
            },
        }
        self._write_log(payload)
        return payload

    def compare_chunking(self, sample_query: str) -> pd.DataFrame:
        records = []
        for strategy in ["fixed", "structure"]:
            bot = RAGChatbot(docs_path=self.docs_path, strategy=strategy, top_k=self.top_k)
            result = bot.query(sample_query)
            score_sum = sum(item["combined_score"] for item in result["retrieved"])
            records.append(
                {
                    "strategy": strategy,
                    "chunks_count": len(bot.chunks),
                    "topk_score_sum": score_sum,
                    "total_latency_ms": result["latency_ms"]["total"],
                }
            )
        return pd.DataFrame(records)

    def compare_failure_case(self, failure_query: str, top_k: int = 4) -> Dict[str, object]:
        cleaned_query = normalize_text(failure_query)
        query_emb = self.embedder.encode([cleaned_query])

        # Vector-only baseline
        vector_only: List[Dict[str, object]] = []
        if self.retriever.use_faiss and self.retriever.index is not None:
            scores, indices = self.retriever.index.search(query_emb, top_k)
            for i, s in zip(indices[0], scores[0]):
                if i < 0:
                    continue
                chunk = self.chunks[int(i)]
                vector_only.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "text": chunk.text,
                        "score": float(s),
                    }
                )
        else:
            sim = cosine_similarity(query_emb, self.embeddings)[0]
            rank = np.argsort(sim)[::-1][:top_k]
            for i in rank:
                chunk = self.chunks[int(i)]
                vector_only.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "text": chunk.text,
                        "score": float(sim[int(i)]),
                    }
                )

        hybrid = self.retriever.search(cleaned_query, query_emb, top_k=top_k)
        result = {
            "query": cleaned_query,
            "vector_only": vector_only,
            "hybrid": hybrid,
        }
        self._write_experiment({"type": "failure_case_compare", "timestamp": datetime.utcnow().isoformat(), **result})
        return result

    def run_adversarial_suite(self) -> Dict[str, object]:
        test_queries = [
            {
                "name": "ambiguous_query",
                "query": "What was the best performance in 2024?",
                "risk": "ambiguity",
            },
            {
                "name": "misleading_query",
                "query": "Budget says debt is fully solved, right?",
                "risk": "misleading premise",
            },
        ]
        runs: List[Dict[str, object]] = []
        for item in test_queries:
            out = self.query(item["query"], include_pure_baseline=True)
            runs.append(
                {
                    "name": item["name"],
                    "risk": item["risk"],
                    "query": item["query"],
                    "rag_answer": out["answer"],
                    "pure_llm_answer": out["pure_llm_answer"],
                    "retrieved": out["retrieved"],
                }
            )

        metrics = self.compute_metrics(runs)
        payload = {"timestamp": datetime.utcnow().isoformat(), "runs": runs, "metrics": metrics}
        self._write_experiment({"type": "adversarial_suite", **payload})
        return payload

    def compute_metrics(self, runs: List[Dict[str, object]]) -> Dict[str, float]:
        grounded = 0
        hallucination_flags = 0
        consistencies: List[float] = []

        for run in runs:
            rag = run["rag_answer"]
            pure = run["pure_llm_answer"]
            retrieved_text = " ".join(item["text"] for item in run["retrieved"])
            has_citation = "[" in rag and "]" in rag
            overlap = _jaccard(rag, retrieved_text)
            if has_citation or overlap > 0.15:
                grounded += 1
            if overlap < 0.08:
                hallucination_flags += 1
            consistencies.append(_jaccard(rag, pure))

        n = max(1, len(runs))
        return {
            "accuracy_proxy": round(grounded / n, 3),
            "groundedness": round(grounded / n, 3),
            "hallucination_rate": round(hallucination_flags / n, 3),
            "response_consistency_proxy": round(sum(consistencies) / n, 3),
        }

    def save_feedback(self, query: str, answer: str, rating: str, note: str = "") -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "rating": rating,
            "note": note,
            "answer_preview": answer[:300],
        }
        with self.feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _write_log(self, payload: Dict[str, object]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _write_experiment(self, payload: Dict[str, object]) -> None:
        with self.experiment_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
