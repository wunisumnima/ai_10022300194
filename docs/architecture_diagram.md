# RAG Architecture Diagram

```mermaid
flowchart LR
    A[User Query] --> B[Query Cleaner]
    B --> C[Embedding Pipeline]
    C --> D[Hybrid Retriever]
    D --> D1[Vector Similarity FAISS]
    D --> D2[Keyword Similarity TF-IDF]
    D1 --> E[Top-k Context Selector]
    D2 --> E
    E --> F[Prompt Builder]
    F --> G[LLM Generator]
    G --> H[Final Response]

    E --> I[Retrieved Chunks + Scores UI]
    F --> J[Prompt Transparency UI]
    H --> K[Feedback Loop]
    K --> L[feedback.jsonl]
    H --> M[runs.jsonl]
    N[Adversarial Test Runner] --> O[experiments.jsonl]
```

## Notes

- Retrieval fix path: hybrid ranking (vector + keyword + domain boost)
- Evaluation path: adversarial queries + RAG vs pure LLM comparison
- Logging path: run logs, experiments, and feedback are stored in `logs/`
