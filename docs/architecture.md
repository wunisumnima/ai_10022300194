# RAG Architecture Notes

Name: `Sumnima Wuni`
Index Number: `10022300194`

## Architecture Components

1. Data Ingestion Layer
- Reads PDF, CSV, TXT, and MD files from `docs/`.

2. Cleaning and Normalization Layer
- Removes noisy whitespace and normalizes text.

3. Chunking Layer
- Strategy A: fixed-size windows with overlap.
- Strategy B: structure-aware chunks for heading-based segmentation.

4. Embedding Layer
- Local hashing embeddings (scikit-learn `HashingVectorizer`).

4b. Election aggregate (optional)
- For vote/election-style questions, national totals from `Ghana_Election_Result.csv` are computed and prepended to retrieved context.

5. Vector Store and Retrieval Layer
- FAISS index for vector retrieval.
- TF-IDF keyword scoring for lexical relevance.
- Hybrid ranking combines vector and keyword scores.

6. Context Selection Layer
- Selects top-k ranked chunks.
- Truncates context to fit prompt budget.

7. Prompt Engineering Layer
- Injects retrieved evidence.
- Enforces hallucination control:
  - answer from context only
  - abstain when not found
  - include citations

8. Generation Layer
- Groq OpenAI-compatible API only (`config.py` / `GROQ_API_KEY`).
- Includes pure LLM baseline for comparison.

9. Logging and Evaluation Layer
- Logs retrieval, scores, prompt, answer, and latency to `logs/runs.jsonl`.
- Supports adversarial query evaluation and strategy comparison.

10. User Interface Layer
- Streamlit app for input, retrieval display, prompt transparency, and output comparison.

## Data Flow

User query -> Embed query -> Retrieve top-k (hybrid) -> Build grounded prompt -> Generate answer -> Display answer and evidence -> Log run

## Why This Design Fits the Domain

- Budget document is long and structured; structure-aware chunking helps preserve section meaning.
- Election dataset is tabular; row-level chunking improves factual retrieval.
- Hybrid retrieval improves acronym and exact-term handling common in policy/economic documents.
- Prompt constraints reduce hallucinations and improve traceable answers.
