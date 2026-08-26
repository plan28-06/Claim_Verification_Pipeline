# Scientific_Claim_Verification_Pipeline using NLI + RAG

A hybrid **scientific claim verification system** that verifies claims against research papers using **Natural Language Inference (NLI)** and **Retrieval-Augmented Generation (RAG)**.

The system retrieves the most relevant evidence from a research paper, independently verifies the claim using a DeBERTa NLI model and a local LLM, and combines both predictions into a final verdict.

## Overview

Given a research paper and a claim such as:

> "Sol–gel silica coating increased the fracture resistance of annealed soda-lime-silica glass by approximately 35 MPa."

the system determines whether the claim is:

- **SUPPORTED**
- **CONTRADICTED**
- **NOT ENOUGH EVIDENCE**

The verification uses two independent approaches:

1. **NLI-based verification**
2. **RAG-based LLM verification**

The final verdict is produced only when both approaches agree.

---

## System Architecture

```text
                         Research Paper
                               |
                               v
                         PDF Ingestion
                               |
                               v
                         Text Cleaning
                               |
                               v
                           Chunking
                               |
                               v
                            Claim
                               |
                               v
                    Hybrid Retrieval
                   (Semantic + BM25)
                               |
                         Top 5 Chunks
                         /           \
                        /             \
                       v               v
                NLI Verification    RAG / LLM
                 (DeBERTa-v3)       Verification
                       |               |
                       v               v
                 NLI Verdict       LLM Verdict
                       |               |
                       \───────┬───────/
                               |
                               v
                       Agreement Check
                               |
                               v
                       Final Verdict
```

## 1. PDF Ingestion

The research paper is processed through an ingestion pipeline:

```text
PDF
 |
 v
Text Extraction
 |
 v
Text Cleaning
 |
 v
Paragraph Chunking
```

The ingestion pipeline:

- extracts text from the PDF using **PyMuPDF**
- removes common PDF extraction artifacts
- fixes words split across lines
- removes repeated headers/footers
- splits the document into manageable chunks

---

## 2. Hybrid Retrieval

For each claim, the system retrieves the most relevant evidence chunks.

Two retrieval approaches are combined.

### Semantic Retrieval

Uses **`all-MiniLM-L6-v2`** to get vector embeddings of claim and document chunk after which cosine similarity is computed.

### Lexical Retrieval

Uses **BM25** to measure lexical overlap between the claim and document chunks.

Lexical approach is used as semantic retrieval is not great with numbers.

The scores are normalized and combined:

```text
Combined Score =
    semantic_weight × semantic_score
    +
    (1 - semantic_weight) × BM25_score
```
semantic_weight = 0.5

The highest-ranking chunks are selected as evidence.

Currently:

```text
Top K = 5
```

---

# 3. NLI Verification

The first verification stage uses:

**`DeBERTa-v3-base-mnli-fever-anli`**

For every retrieved chunk, the model receives:

```text
Premise    = Research paper chunk
Hypothesis = Claim
```

and produces three probabilities:

```text
Entailment
Contradiction
Neutral
```

For each chunk:

```text
Entailment + Contradiction + Neutral = 1
```

### Relevance-weighted aggregation

The NLI probabilities are weighted using the retrieval relevance of each chunk:

```text
weighted_support =
    relevance × entailment_probability

weighted_contradiction =
    relevance × contradiction_probability

weighted_neutral =
    relevance × neutral_probability
```

These values are then summed across all retrieved chunks:

```text
total_support_score
total_contradiction_score
total_neutral_score
```

### NLI Verdict

The system compares support and contradiction while treating neutral evidence separately.

```text
non_neutral =
    total_support_score
    +
    total_contradiction_score
```

Then:

```text
support_ratio =
    total_support_score / non_neutral

contradiction_ratio =
    total_contradiction_score / non_neutral
```

The current decision threshold is **0.65**:

```text
support_ratio >= 0.65
        -> SUPPORTED

contradiction_ratio >= 0.65
        -> CONTRADICTED

otherwise
        -> NOT ENOUGH EVIDENCE
```

---



### Why Neutral Evidence Is Excluded from the Support/Contradiction Ratio

Neutral means:

> "This chunk doesn't provide enough information to say whether the claim is true or false."

It is **not evidence against the claim**, and it is **not evidence for the claim**.

Therefore, we do not want a large number of neutral chunks to drown out the actual support/contradiction evidence.

For example, suppose after aggregating 5 chunks:

```text
Support       = 0.80
Contradiction = 0.10
Neutral       = 4.10
```

If neutral were included in the denominator:

```text
Support ratio = 0.80 / (0.80 + 0.10 + 4.10)
              = 16%
```

This would make the support look artificially weak.

Instead, we compare only the **non-neutral evidence**:

```text
Non-neutral = Support + Contradiction
            = 0.80 + 0.10
            = 0.90
```

Therefore:

```text
Support ratio =
    0.80 / 0.90
    = 88.9%

Contradiction ratio =
    0.10 / 0.90
    = 11.1%
```

The claim is therefore classified as **SUPPORTED**.

Neutral scores are still retained and reported because they indicate how much of the retrieved evidence is inconclusive. They are simply not allowed to dominate the directional comparison between support and contradiction.

# 4. RAG-Based LLM Verification

The second verification stage uses a local LLM (llama 3.1 8b).

Instead of providing the entire research paper to the LLM, the system retrieves the **top 5 relevant chunks** and provides them as context along with their relevance scores.

```text
Claim
+
Retrieved Evidence
        |
        v
      Local LLM
        |
        v
SUPPORTED /
CONTRADICTED /
NOT ENOUGH EVIDENCE
```

The LLM is instructed to:

- use only the supplied paper evidence
- avoid outside knowledge
- distinguish support from contradiction
- identify when evidence is insufficient
- pay attention to numerical values, units, conditions, and qualifiers

This constitutes a **Retrieval-Augmented Generation (RAG)** approach to claim verification.

---

# 5. Final Verification

The NLI and LLM operate independently.

Their outputs are compared:

| NLI | LLM | Final |
|---|---|---|
| SUPPORTED | SUPPORTED | **SUPPORTED** |
| CONTRADICTED | CONTRADICTED | **CONTRADICTED** |
| NOT ENOUGH EVIDENCE | NOT ENOUGH EVIDENCE | **NOT ENOUGH EVIDENCE** |
| SUPPORTED | CONTRADICTED | **NOT ENOUGH EVIDENCE** |
| SUPPORTED | NOT ENOUGH EVIDENCE | **NOT ENOUGH EVIDENCE** |
| CONTRADICTED | SUPPORTED | **NOT ENOUGH EVIDENCE** |
| CONTRADICTED | NOT ENOUGH EVIDENCE | **NOT ENOUGH EVIDENCE** |

The system therefore uses a **conservative agreement-based decision rule**.

If the two independent verification methods disagree, the system does not force a binary decision and outputs NOT ENOUGH EVIDENCE.

---


## Running the System

Set the PDF path and claims in `main.py`:

```python
PDF_PATH = r"path/to/research_paper.pdf"

CLAIMS = [
    "Your first claim.",
    "Your second claim.",
    "Your third claim."
]
```

```bash
ollama serve
```

Then run:

```bash
python main.py
```
