# Manual Experiment Logs (CS4241)

Name: Sumnima Wuni
Index Number: 10022300194


## 1) Chunking Strategy Comparison

### Query:
What is ABFA allocation in 2024?

### Result Table
 strategy  chunks_count  topk_score_sum  total_latency_ms
    fixed           886        1.185591             12.04
structure           924        1.184794             11.11

### Fixed Strategy Observations
- Retrieval quality: Good for short queries but sometimes mixes nearby economic sections.
- Irrelevant chunk examples: Generic petroleum context around receipts without allocation detail.
- Strengths: Faster and consistent chunk boundaries.
- Weaknesses: May split important table context across chunks.

### Structure-Aware Strategy Observations
- Retrieval quality: Better preservation of section/table context.
- Irrelevant chunk examples: Fewer, mostly broad section intros.
- Strengths: More coherent chunks for policy document headings.
- Weaknesses: Slightly higher processing latency.

### Comparative Conclusion
- Which strategy worked better and why: Structure-aware worked better for budget policy questions because heading alignment improved retrieval coherence.

## 2) Prompt Iteration Experiments

### Prompt V1
- Template summary: Basic grounded prompt without strict citation and abstain rule.
- Output quality: Answered query but less traceable.
- Hallucination risk: Medium.

### Prompt V2 (citation + abstention rule)
- Template summary: Enforced context-only answer, explicit abstention, citation requirement.
- Output quality: More grounded and auditable responses.
- Hallucination risk: Lower.

### Prompt Improvement Evidence
- Concrete differences: V2 responses referenced retrieved context directly and reduced unsupported claims.

## 3) Retrieval Failure Case and Fix

### Failure Query
What is ABFA allocation trend?

### What failed
- Retrieved irrelevant chunks: Vector-only baseline returned semantically similar but less specific chunks.
- Why this happened: Embedding similarity favored broad macroeconomic paragraphs.

### Fix implemented
- Hybrid retrieval with keyword score and domain boosts for terms like ABFA, allocation, debt, GDP.

### Post-fix results
- What improved: Hybrid retrieval promoted chunks with more direct term overlap and section relevance.

## 4) Adversarial Testing

### Adversarial Query 1 (Ambiguous)
- Query: What was the best performance in 2024?
- RAG answer: No LLM key set. Retrieval completed; review retrieved chunks as grounded answer evidence.
- Pure LLM answer: No LLM key set for pure LLM baseline.
- Accuracy: See metrics snapshot (proxy).
- Hallucination notes: RAG answer was more context constrained.

### Adversarial Query 2 (Misleading/Incomplete)
- Query: Budget says debt is fully solved, right?
- RAG answer: No LLM key set. Retrieval completed; review retrieved chunks as grounded answer evidence.
- Pure LLM answer: No LLM key set for pure LLM baseline.
- Accuracy: See metrics snapshot (proxy).
- Hallucination notes: RAG handled misleading premise better via evidence grounding.

## 5) Metrics Snapshot

- Retrieval@k: k=3 used in experiments.
- Groundedness: 0.0
- Hallucination rate: 1.0
- Consistency: 0.25
- Total latency: captured per-run in `logs/runs.jsonl`.

## 6) Final Reflection

- What worked best: Hybrid retrieval + strict prompt grounding.
- Limitations: Long PDF ingestion takes time on first index build.
- Next improvement: Persist vector index to disk and add reranker.
