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
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
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


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: str(datetime.now()))


@dataclass
class Experiment:
    id: str
    name: str
    description: str
    parameters: Dict[str, any]
    results: Dict[str, any]
    created_at: str = field(default_factory=lambda: str(datetime.now()))
    status: str = "active"  # active, completed, failed


@dataclass
class ManualLog:
    timestamp: str
    level: str  # INFO, WARNING, ERROR, DEBUG
    message: str
    category: str = "general"
    experiment_id: Optional[str] = None


class LogManager:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logs: List[ManualLog] = []
        self.setup_file_logging()
        
    def setup_file_logging(self):
        """Setup file logging for the application"""
        log_file = self.log_dir / f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def add_log(self, level: str, message: str, category: str = "general", experiment_id: Optional[str] = None):
        """Add a manual log entry"""
        log_entry = ManualLog(
            timestamp=str(datetime.now()),
            level=level.upper(),
            message=message,
            category=category,
            experiment_id=experiment_id
        )
        self.logs.append(log_entry)
        
        # Also log to file
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(f"[{category}] {message}")
        
        # Save to JSON file
        self.save_logs_to_file()
    
    def save_logs_to_file(self):
        """Save logs to JSON file"""
        logs_file = self.log_dir / "manual_logs.json"
        logs_data = [log.__dict__ for log in self.logs]
        with open(logs_file, 'w', encoding='utf-8') as f:
            json.dump(logs_data, f, indent=2, ensure_ascii=False)
    
    def get_logs(self, category: Optional[str] = None, experiment_id: Optional[str] = None) -> List[ManualLog]:
        """Get filtered logs"""
        filtered_logs = self.logs
        if category:
            filtered_logs = [log for log in filtered_logs if log.category == category]
        if experiment_id:
            filtered_logs = [log for log in filtered_logs if log.experiment_id == experiment_id]
        return filtered_logs
    
    def clear_logs(self):
        """Clear all logs"""
        self.logs = []
        self.save_logs_to_file()


class ExperimentManager:
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        self.experiments: Dict[str, Experiment] = {}
        self.load_experiments()
    
    def create_experiment(self, name: str, description: str, parameters: Dict[str, any]) -> str:
        """Create a new experiment"""
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.experiments)}"
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            parameters=parameters,
            results={}
        )
        self.experiments[experiment_id] = experiment
        self.save_experiments()
        return experiment_id
    
    def update_experiment_results(self, experiment_id: str, results: Dict[str, any]):
        """Update experiment results"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id].results.update(results)
            self.save_experiments()
    
    def complete_experiment(self, experiment_id: str, status: str = "completed"):
        """Mark experiment as completed"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id].status = status
            self.save_experiments()
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        return self.experiments.get(experiment_id)
    
    def list_experiments(self) -> List[Experiment]:
        """List all experiments"""
        return list(self.experiments.values())
    
    def save_experiments(self):
        """Save experiments to JSON file"""
        experiments_file = self.experiments_dir / "experiments.json"
        experiments_data = {exp_id: exp.__dict__ for exp_id, exp in self.experiments.items()}
        with open(experiments_file, 'w', encoding='utf-8') as f:
            json.dump(experiments_data, f, indent=2, ensure_ascii=False)
    
    def load_experiments(self):
        """Load experiments from JSON file"""
        experiments_file = self.experiments_dir / "experiments.json"
        if experiments_file.exists():
            with open(experiments_file, 'r', encoding='utf-8') as f:
                experiments_data = json.load(f)
                for exp_id, exp_data in experiments_data.items():
                    self.experiments[exp_id] = Experiment(**exp_data)


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
        self.chat_history: List[ChatMessage] = []
        self.log_manager = LogManager()
        self.experiment_manager = ExperimentManager()
        self.current_experiment_id: Optional[str] = None
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.load_chat_history()
        self.log_manager.add_log("INFO", f"Chatbot initialized with session ID: {self.session_id}", "system")

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

    def load_chat_history(self) -> None:
        """Load chat history from file"""
        history_file = Path("chat_history.json")
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                self.chat_history = [ChatMessage(**msg) for msg in history_data]
                self.log_manager.add_log("INFO", f"Loaded {len(self.chat_history)} messages from history", "chat")

    def save_chat_history(self) -> None:
        """Save chat history to file"""
        history_file = Path("chat_history.json")
        history_data = [msg.__dict__ for msg in self.chat_history]
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)

    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to chat history"""
        self.chat_history.append(ChatMessage(role=role, content=content))
        # Keep only last 50 exchanges (100 messages) to manage memory
        if len(self.chat_history) > 100:
            self.chat_history = self.chat_history[-100:]
        self.save_chat_history()
        self.log_manager.add_log("INFO", f"Added {role} message to history", "chat")

    def get_history_string(self, max_exchanges: int = 10) -> str:
        """Get formatted chat history for context"""
        if not self.chat_history:
            return ""
        
        recent_history = self.chat_history[-(max_exchanges * 2):]  # Last N exchanges
        history_lines = []
        for msg in recent_history:
            prefix = "User" if msg.role == "user" else "Assistant"
            history_lines.append(f"{prefix}: {msg.content}")
        
        return "\n".join(history_lines)

    def clear_history(self) -> None:
        """Clear chat history"""
        self.chat_history = []
        self.save_chat_history()
        self.log_manager.add_log("INFO", "Chat history cleared", "chat")

    def start_experiment(self, name: str, description: str, parameters: Dict[str, any]) -> str:
        """Start a new experiment"""
        experiment_id = self.experiment_manager.create_experiment(name, description, parameters)
        self.current_experiment_id = experiment_id
        self.log_manager.add_log("INFO", f"Started experiment: {name} (ID: {experiment_id})", "experiment", experiment_id)
        return experiment_id

    def log_experiment_result(self, metric: str, value: any):
        """Log experiment result"""
        if self.current_experiment_id:
            self.experiment_manager.update_experiment_results(self.current_experiment_id, {metric: value})
            self.log_manager.add_log("INFO", f"Experiment result: {metric} = {value}", "experiment", self.current_experiment_id)

    def end_experiment(self, status: str = "completed"):
        """End current experiment"""
        if self.current_experiment_id:
            self.experiment_manager.complete_experiment(self.current_experiment_id, status)
            self.log_manager.add_log("INFO", f"Experiment ended with status: {status}", "experiment", self.current_experiment_id)
            self.current_experiment_id = None

    def answer(self, query: str) -> str:
        start_time = datetime.now()
        self.log_manager.add_log("INFO", f"Processing query: {query[:50]}...", "query")
        
        hits = self.retriever.search(query, top_k=self.top_k)
        if not hits:
            self.log_manager.add_log("WARNING", "No relevant documents found", "query")
            return "I could not find relevant documents. Please try a different question."

        context = "\n\n".join(
            f"Source: {chunk.source}\n{chunk.text}"
            for chunk, _score in hits
        )

        history = self.get_history_string()
        
        prompt = (
            "You are a helpful assistant with access to retrieved knowledge from local documents. "
            "Use the context to answer the user question factually. If the answer is not present, say you do not know. "
            "Consider the conversation history for context and to maintain conversation coherence.\n\n"
        )
        
        if history:
            prompt += f"Conversation History:\n{history}\n\n"
        
        prompt += f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        answer = self._generate_text(prompt)
        
        # Log performance metrics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        self.log_manager.add_log("INFO", f"Query processed in {processing_time:.2f} seconds", "query")
        
        # Log experiment results if experiment is active
        if self.current_experiment_id:
            self.log_experiment_result("processing_time", processing_time)
            self.log_experiment_result("retrieved_docs", len(hits))
            self.log_experiment_result("answer_length", len(answer))
        
        return answer

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
    print("\nAvailable commands:")
    print("  Type a question to chat")
    print("  'history' - View conversation history")
    print("  'clear' - Clear conversation history")
    print("  'logs' - View recent logs")
    print("  'log <level> <message>' - Add manual log (level: info, warning, error)")
    print("  'experiment start <name> <description>' - Start new experiment")
    print("  'experiment end [status]' - End current experiment")
    print("  'experiments' - List all experiments")
    print("  'experiment <id>' - View experiment details")
    print("  'exit' - Quit the chatbot")

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break
        elif question.lower() == "history":
            history = bot.get_history_string()
            if history:
                print("\n--- Conversation History ---")
                print(history)
                print("--- End of History ---")
            else:
                print("\nNo conversation history yet.")
            continue
        elif question.lower() == "clear":
            bot.clear_history()
            print("\nConversation history cleared.")
            continue
        elif question.lower() == "logs":
            logs = bot.log_manager.get_logs()[-10:]  # Last 10 logs
            if logs:
                print("\n--- Recent Logs ---")
                for log in logs:
                    print(f"[{log.timestamp}] {log.level} [{log.category}] {log.message}")
                print("--- End of Logs ---")
            else:
                print("\nNo logs yet.")
            continue
        elif question.lower().startswith("log "):
            parts = question[4:].split(" ", 2)
            if len(parts) >= 2:
                level, message = parts[0], parts[1]
                category = parts[2] if len(parts) > 2 else "manual"
                bot.log_manager.add_log(level, message, category)
                print(f"\nLog added: [{level.upper()}] {message}")
            else:
                print("\nUsage: log <level> <message> [category]")
            continue
        elif question.lower().startswith("experiment start "):
            parts = question[17:].split(" ", 1)
            if len(parts) >= 2:
                name, description = parts[0], parts[1]
                experiment_id = bot.start_experiment(name, description, {"session_id": bot.session_id})
                print(f"\nExperiment started: {name} (ID: {experiment_id})")
            else:
                print("\nUsage: experiment start <name> <description>")
            continue
        elif question.lower().startswith("experiment end"):
            parts = question.split(" ", 2)
            status = parts[2] if len(parts) > 2 else "completed"
            if bot.current_experiment_id:
                bot.end_experiment(status)
                print(f"\nExperiment ended with status: {status}")
            else:
                print("\nNo active experiment to end.")
            continue
        elif question.lower() == "experiments":
            experiments = bot.experiment_manager.list_experiments()
            if experiments:
                print("\n--- Experiments ---")
                for exp in experiments:
                    status_icon = "✓" if exp.status == "completed" else "○" if exp.status == "active" else "✗"
                    print(f"{status_icon} {exp.id}: {exp.name} ({exp.status})")
                print("--- End of Experiments ---")
            else:
                print("\nNo experiments yet.")
            continue
        elif question.lower().startswith("experiment "):
            exp_id = question[11:].strip()
            experiment = bot.experiment_manager.get_experiment(exp_id)
            if experiment:
                print(f"\n--- Experiment Details ---")
                print(f"ID: {experiment.id}")
                print(f"Name: {experiment.name}")
                print(f"Description: {experiment.description}")
                print(f"Status: {experiment.status}")
                print(f"Created: {experiment.created_at}")
                print(f"Parameters: {experiment.parameters}")
                print(f"Results: {experiment.results}")
                print("--- End of Details ---")
            else:
                print(f"\nExperiment not found: {exp_id}")
            continue

        # Add user question to history
        bot.add_to_history("user", question)
        
        answer = bot.answer(question)
        print(f"\nBot: {answer}")
        
        # Add bot answer to history
        bot.add_to_history("assistant", answer)


if __name__ == "__main__":
    main()
