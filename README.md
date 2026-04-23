# Academic City RAG Chatbot (CS4241)

Name: `Sumnima Wuni`
Index Number: `10022300194`

Repository naming rule from question paper:
- `ai_<index_number>`

## Project Overview

This project implements a manual Retrieval-Augmented Generation (RAG) chatbot for Academic City data.
The pipeline is built without end-to-end frameworks such as LangChain or LlamaIndex.

Pipeline:
- User Query
- Retrieval (hybrid: vector + keyword)
- Context selection
- Prompt construction
- LLM response

## Data Sources

Place project datasets in `docs/`:
- `Ghana_Election_Result.csv`
- `2025-Budget-Statement-and-Economic-Policy_v4.pdf`

## Implemented Exam Requirements

- Data cleaning and ingestion for CSV/TXT/MD/PDF
- Two chunking strategies:
  - `fixed`: token-like fixed windows with overlap
  - `structure`: heading-aware segmentation and bounded windows
- Local hashing embeddings (no OpenAI)
- Groq-only LLM (`config.py` or `GROQ_API_KEY`)
- Election CSV: optional **national vote aggregates** injected for vote/winner questions
- Vector retrieval with FAISS (fallback supported if FAISS unavailable)
- Top-k retrieval + similarity scoring
- Hybrid retrieval extension (vector + keyword scoring + domain boost)
- Prompt template with hallucination control and citation requirement
- Context window management with max context character budget
- Stage-by-stage logging:
  - retrieved chunks
  - similarity scores
  - final prompt
  - latency by stage
- Adversarial test support and RAG vs pure LLM output comparison
- Built-in failure-case comparison (vector-only vs hybrid retrieval)
- Innovation component: user feedback loop persisted to `logs/feedback.jsonl`
- Streamlit UI:
  - query input
  - retrieved chunks and scores
  - final answer
  - prompt transparency
  - chunking comparison button

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Groq only.** Put your key in `config.py` as `GROQ_API_KEY`, or set:

```powershell
$env:GROQ_API_KEY="gsk_..."
```

Optional: override in the Streamlit app (**Override Groq key**).

## Run

```bash
python -m streamlit run app.py
```

## Manual Experiment Files

- `docs/manual_experiment_logs.md`
- `docs/architecture.md`
- `docs/architecture_diagram.md`

Fill these with your own experiment observations (manual, not AI-generated summaries).

## Required Submission Notes

- Add your actual name/index in this README and in your source files if instructed by lecturer.
- Push code to GitHub repository named `ai_<index_number>`.
- Deploy app and include deployment URL.
- Invite lecturer as collaborator:
  - `godwin.danso@acity.edu.gh` or `GodwinDansoAcity`
- Submit links and documentation by email as instructed.
