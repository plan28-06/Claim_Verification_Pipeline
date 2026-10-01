# Claim Verification Pipeline

A hybrid retrieval-based **scientific claim verification system** that verifies claims against research papers using **hybrid information retrieval, Natural Language Inference (NLI), and open-source LLMs**.

The system retrieves the most relevant evidence from a research paper and independently verifies a claim using:

1. **NLI-based verification** using DeBERTa
2. **Retrieval-based verification** using local open-source LLMs

The project also contains an experimental evaluation framework for comparing different verification approaches on a constructed scientific claim verification benchmark.

---

# Overview

Given a research paper and a claim such as:

> "Sol–gel silica coating increased the fracture resistance of annealed soda-lime-silica glass by approximately 35 MPa."

the system determines whether the claim is:

- **SUPPORTED**
- **CONTRADICTED**
- **NOT ENOUGH EVIDENCE**

The overall pipeline is:

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
      v
Top 5 Evidence
      |
      +-------------------+
      |                   |
      v                   v
 NLI Verification    RAG / LLM Verification
   DeBERTa            Llama / Qwen / Mistral
      |                   |
      v                   v
 NLI Verdict          LLM Verdict
      |                   |
      +---------+---------+
                |
                v
        Agreement Check
                |
                v
         Final Verdict
```

---

# Project Structure

The repository is divided into two major folders:

```text
Claim_Verification_Pipeline/
│
├── Paper_Pipeline/
│   └── ...
│
└── Test/
    ├── data/
    │   ├── scifact/
    │   ├── clinifact/
    │   └── constructed_dataset/
    │
    ├── generate_retrieval_cache.py
    ├── retrieval_cache.json
    ├── nli_verify.py
    ├── run_classification.py
    │
    ├── llama/
    │   └── llm_verify_test_llama.py
    │
    ├── qwen/
    │   └── llm_verify_test_qwen.py
    │
    ├── mistral/
    │   └── llm_verify_test_mistral.py
    │
    └── results/
```

## `Paper_Pipeline`

The `Paper_Pipeline` folder contains the **paper processing pipeline**.

It is responsible for taking a research paper in PDF format and converting it into clean textual evidence that can be retrieved during claim verification.

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
Chunking
 |
 v
Evidence
```

This part of the repository represents the actual paper-processing component of the system.

---

## `Test`

The `Test` folder contains the **experimental evaluation framework**.

It contains:

- benchmark datasets
- dataset construction
- frozen retrieval results
- NLI verification
- LLM verification
- model comparison
- classification evaluation
- experimental results

The separation allows the paper-processing pipeline to remain independent from the benchmark and experimental setup.

---

# 1. PDF Ingestion

Research papers are processed using **PyMuPDF**.

The ingestion pipeline performs:

- PDF text extraction
- removal of common PDF extraction artifacts
- correction of words split across lines
- removal of repeated headers and footers
- text cleaning
- paragraph/sentence chunking

The goal is to convert the raw PDF into clean textual evidence that can be searched and passed to the verification models.

---

# 2. Dataset Construction

The experimental benchmark was constructed using:

- **SciFact**
- **CliniFact**

The goal was to create a three-class scientific claim verification benchmark containing:

```text
SUPPORTED
CONTRADICTED
NOT ENOUGH EVIDENCE
```

No NLI or LLM model was fine-tuned on this dataset.

---

# 3. SciFact Processing

SciFact was used for claims with explicit evidence annotations.

The following splits were used:

```text
SciFact TRAIN → classification benchmark
SciFact DEV   → validation and method selection
SciFact TEST  → not used
```

## SciFact TRAIN

The original training set contained:

```text
809 claims
```

We removed:

```text
77 multi-document claims
284 no-evidence claims
```

This produced:

```text
448 usable claims
```

with:

```text
SUPPORTED       = 296
CONTRADICTED    = 152
```

Multi-document claims were excluded because the verification pipeline evaluates claims against evidence retrieved from a single research paper.

No-evidence claims were excluded because they did not provide the evidence required for the binary SciFact verification setup.

---

# 4. SciFact Development Set

The SciFact development set was used only for **validation and methodology selection**.

The original development set contained:

```text
300 claims
```

After filtering:

```text
24 multi-document claims
110 no-evidence claims
```

were removed.

The resulting validation set contained:

```text
166 claims

SUPPORTED       = 108
CONTRADICTED    = 58
```

The filtered development set contains no gold `NOT ENOUGH EVIDENCE` examples.

Therefore, NLI aggregation and threshold selection were performed using **binary Macro-F1 over SUPPORTED and CONTRADICTED claims**.

The final 2,035-claim benchmark was not used for tuning.

---

# 5. CliniFact Processing

CliniFact was used to introduce the **NOT ENOUGH EVIDENCE** class.

The original dataset contained:

```text
1970 claims
```

The labels were processed as:

```text
Inconclusive
Evidence
Not Enough Evidence
```

The `Inconclusive` class was excluded.

The retained examples were:

```text
Evidence              = 562
Not Enough Evidence   = 1025
```

The Evidence class was mapped to:

```text
SUPPORTED
```

Therefore, CliniFact contributed:

```text
SUPPORTED             = 562
NOT ENOUGH EVIDENCE   = 1025
```

---

# 6. Final Constructed Dataset

The final benchmark combines the usable SciFact and CliniFact records.

```text
SciFact
    SUPPORTED       = 296
    CONTRADICTED    = 152

CliniFact
    SUPPORTED       = 562
    NOT ENOUGH      = 1025
```

Final dataset:

```text
Total = 2035 claims

SUPPORTED             = 858
CONTRADICTED          = 152
NOT ENOUGH EVIDENCE   = 1025
```

The final dataset is therefore a three-class classification benchmark.

> **Important:** Because of the way the datasets are combined, `CONTRADICTED` examples come from SciFact while `NOT ENOUGH EVIDENCE` examples come from CliniFact. This source/class association is considered a limitation of the constructed benchmark.

---

# 7. Hybrid Retrieval

For every claim, the system retrieves the most relevant evidence from the corresponding research paper.

Two retrieval methods are combined:

1. **Semantic retrieval**
2. **BM25 lexical retrieval**

---

## Semantic Retrieval

Semantic retrieval uses:

```text
all-MiniLM-L6-v2
```

The claim and paper sentences are converted into embeddings.

Cosine similarity is then calculated between the claim and each sentence.

Semantic retrieval helps identify evidence that is semantically related to the claim even when different terminology is used.

---

## BM25 Retrieval

BM25 is used to measure lexical overlap between the claim and paper sentences.

BM25 is particularly useful for:

- scientific terminology
- exact phrases
- names
- numerical information
- technical terms

---

# 8. Hybrid Retrieval Score

The semantic and BM25 scores are normalized and combined:

```text
Combined Score =
    semantic_weight × semantic_score
    +
    BM25_weight × BM25_score
```

The retrieval weights were tuned on the filtered SciFact development set.

Several combinations were evaluated using:

- Recall@1
- Recall@3
- Recall@5
- MRR

The selected configuration was:

```text
Semantic weight = 0.9
BM25 weight     = 0.1
```

The development retrieval results were:

```text
Recall@1 = 0.5602
Recall@3 = 0.8434
Recall@5 = 0.9398
MRR      = 0.7097
```

This retrieval configuration was then frozen for the final classification experiments.

---

# 9. Evidence Selection

For every claim:

```text
Top K = 5
```

evidence sentences are retrieved.

Each retrieved item contains:

```text
Sentence
Sentence index
Retrieval relevance score
```

The verification models receive these retrieved sentences rather than the entire research paper.

---

# 10. Retrieval Cache

To ensure that every verification model receives exactly the same evidence, retrieval is performed once and stored in:

```text
Test/retrieval_cache.json
```

The cache contains the top-5 retrieved evidence for all 2,035 benchmark claims.

Example:

```json
{
    "id": "...",
    "source": "SciFact",
    "claim": "...",
    "gold_label": "SUPPORTED",
    "retrieved_evidence": [
        {
            "sentence_index": 12,
            "sentence": "...",
            "relevance_score": 0.82
        }
    ]
}
```

The gold evidence annotations are not provided to the verification models.

They are used only for retrieval evaluation.

This ensures that Llama, Qwen, Mistral, and DeBERTa are evaluated using the **same frozen retrieved evidence**.

---

# 11. NLI Verification

The NLI stage uses:

```text
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

For every retrieved sentence, the model receives:

```text
Premise:
Research paper evidence

Hypothesis:
Claim
```

The model produces probabilities for:

```text
Entailment
Contradiction
Neutral
```

For each evidence sentence:

```text
Entailment
+
Contradiction
+
Neutral
=
1
```

---

# 12. Relevance-Weighted NLI Aggregation

The NLI probabilities are weighted using the retrieval relevance of each evidence sentence.

For each retrieved sentence:

```text
weighted_support =
    retrieval_relevance × entailment_probability

weighted_contradiction =
    retrieval_relevance × contradiction_probability

weighted_neutral =
    retrieval_relevance × neutral_probability
```

The scores are summed across the top-5 evidence:

```text
total_support_score
total_contradiction_score
total_neutral_score
```

---

# 13. NLI Decision Rule

Neutral evidence is not treated as directional evidence.

Therefore:

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

The NLI threshold was selected using the SciFact development set.

The final threshold is:

```text
0.52
```

The decision rule is:

```text
support_ratio >= 0.52
        → SUPPORTED

contradiction_ratio >= 0.52
        → CONTRADICTED

otherwise
        → NOT ENOUGH EVIDENCE
```

Neutral scores are still retained and reported, but they are excluded from the directional support-vs-contradiction ratio.

---

# 14. Why Neutral Is Excluded

Neutral means that an evidence chunk does not provide enough information to establish either support or contradiction.

For example:

```text
Support       = 0.80
Contradiction = 0.10
Neutral       = 4.10
```

If neutral were included:

```text
Support ratio =
0.80 / (0.80 + 0.10 + 4.10)
= 16%
```

This would make the directional evidence appear artificially weak.

Instead:

```text
Non-neutral =
0.80 + 0.10
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

The claim is classified as:

```text
SUPPORTED
```

---

# 15. NLI Aggregation Experiment

Before freezing the NLI approach, three aggregation strategies were compared on the 166 usable SciFact DEV claims.

All methods used:

```text
Semantic/BM25 = 0.9/0.1
Top-K = 5
DeBERTa-v3
```

The approaches were:

### Relevance-weighted non-neutral

Aggregates NLI probabilities using retrieval relevance.

### Max Evidence

Uses the strongest entailment or contradiction signal among the retrieved evidence.

### Top-1

Uses only the highest-ranked retrieved sentence.

Results:

| Method | Threshold | Macro-F1 |
|---|---:|---:|
| **Relevance-weighted non-neutral** | **0.52** | **0.7240** |
| Max Evidence | 0.60 | 0.6450 |
| Top-1 | 0.50 | 0.5710 |

The relevance-weighted non-neutral approach was therefore frozen for the final experiment.

---

# 16. GPU-Batched NLI Inference

The final benchmark contains:

```text
2035 claims × 5 evidence sentences
= 10,175 NLI pairs
```

The NLI implementation was optimized using:

- GPU inference
- batched tokenization
- batched model inference
- `torch.inference_mode()`

The five evidence sentences are still processed as **five independent NLI pairs**.

They are simply processed together in a batch:

```text
Claim + Evidence 1 ─┐
Claim + Evidence 2 ─┤
Claim + Evidence 3 ─┼──> Batched DeBERTa inference
Claim + Evidence 4 ─┤
Claim + Evidence 5 ─┘
```

The resulting probabilities are then aggregated using the same frozen methodology.

---

# 17. RAG-Based LLM Verification

The second verification stage uses local open-source LLMs.

The evaluated models are:

```text
Llama 3.1 8B
Qwen3 8B
Mistral 7B
```

The models are run locally using **Ollama**.

No model is fine-tuned.

All models receive:

- the same claim
- the same five retrieved evidence sentences
- the same retrieval scores
- the same task definition
- the same output format
- temperature = 0

Only the underlying LLM is changed.

---

# 18. LLM Prompt

The LLM receives the claim and the five retrieved evidence sentences.

Conceptually:

```text
CLAIM:
...

EVIDENCE FROM THE RESEARCH PAPER:

--- EVIDENCE 1 ---
Sentence index: ...
Retrieval relevance: ...

...

--- EVIDENCE 5 ---
Sentence index: ...
Retrieval relevance: ...
```

The LLM is instructed to:

- use only the supplied evidence
- avoid outside knowledge
- distinguish support from contradiction
- identify insufficient evidence
- pay attention to numerical values
- consider units and experimental conditions
- consider qualifiers and comparisons

The required output is:

```json
{
    "verdict": "SUPPORTED | CONTRADICTED | NOT ENOUGH EVIDENCE"
}
```

---

# 19. Final Hybrid Verification

The NLI and LLM operate independently.

Their predictions are then compared.

| NLI | LLM | Final |
|---|---|---|
| SUPPORTED | SUPPORTED | **SUPPORTED** |
| CONTRADICTED | CONTRADICTED | **CONTRADICTED** |
| NEI | NEI | **NEI** |
| SUPPORTED | CONTRADICTED | **NEI** |
| SUPPORTED | NEI | **NEI** |
| CONTRADICTED | SUPPORTED | **NEI** |
| CONTRADICTED | NEI | **NEI** |
| NEI | SUPPORTED | **NEI** |
| NEI | CONTRADICTED | **NEI** |

The final rule is therefore:

```text
if NLI == LLM:
    final = agreed verdict
else:
    final = NOT ENOUGH EVIDENCE
```

This is a **conservative agreement-based fusion strategy**.

---

# 20. Experimental Evaluation

The final benchmark contains:

```text
2035 claims
```

The following systems are evaluated:

```text
1. DeBERTa NLI
2. Llama 3.1 8B
3. Qwen3 8B
4. Mistral 7B

5. NLI + Llama
6. NLI + Qwen
7. NLI + Mistral
```

The primary evaluation metric is **Macro-F1** because of the class imbalance.

We additionally report:

- Accuracy
- Macro Precision
- Macro Recall
- Per-class Precision
- Per-class Recall
- Per-class F1
- Confusion matrices

---

# 21. Final Results

Results on all 2,035 claims:

| System | Accuracy | Macro Precision | Macro Recall | Macro-F1 |
|---|---:|---:|---:|---:|
| DeBERTa NLI | 0.3744 | 0.4645 | 0.5367 | 0.3157 |
| Mistral 7B | 0.7715 | 0.6523 | 0.6319 | 0.6388 |
| Llama 3.1 8B | 0.8113 | 0.7987 | 0.6874 | 0.7204 |
| **Qwen3 8B** | **0.8079** | **0.8207** | **0.7948** | **0.7937** |
| NLI + Mistral | 0.7219 | 0.6490 | 0.5824 | 0.5991 |
| NLI + Llama | 0.7543 | 0.7981 | 0.6369 | 0.6730 |
| NLI + Qwen | 0.7479 | 0.8234 | 0.7233 | 0.7392 |

The results show that the LLM-based approaches outperform the zero-shot DeBERTa NLI baseline on this constructed benchmark.

Qwen3 8B achieves the highest standalone Macro-F1 of **0.7937**.

The agreement-based hybrid does not outperform the corresponding standalone LLM models.

---

# 22. Per-Class Results

## DeBERTa NLI

```text
SUPPORTED
Precision = 0.5561
Recall    = 0.6760
F1        = 0.6102

CONTRADICTED
Precision = 0.1461
Recall    = 0.8882
F1        = 0.2509

NOT ENOUGH EVIDENCE
Precision = 0.6912
Recall    = 0.0459
F1        = 0.0860
```

The particularly low NEI recall indicates that the NLI model frequently assigns a directional verdict to claims that should be classified as insufficiently supported.

---

## Qwen3 8B

```text
SUPPORTED
Precision = 0.9359
Recall    = 0.6294
F1        = 0.7526

CONTRADICTED
Precision = 0.7707
Recall    = 0.7961
F1        = 0.7832

NOT ENOUGH EVIDENCE
Precision = 0.7556
Recall    = 0.9590
F1        = 0.8452
```

Qwen shows strong performance across all three classes on the constructed benchmark.

---

# 23. NLI–LLM Agreement

The verification systems frequently disagree despite receiving the same retrieved evidence.

Agreement rates:

```text
NLI + Llama

Agreement     = 32.97%
Disagreement  = 67.03%
```

```text
NLI + Qwen

Agreement     = 30.81%
Disagreement  = 69.19%
```

```text
NLI + Mistral

Agreement     = 38.33%
Disagreement  = 61.67%
```

The high disagreement rate demonstrates that the NLI and LLM approaches can interpret the same retrieved evidence differently.

It also explains why conservative agreement-based fusion can reduce performance: whenever the two models disagree, the hybrid automatically outputs `NOT ENOUGH EVIDENCE`.

---

# 24. Error Analysis

The system exhibits several recurring error patterns.

### NLI overpredicting directional labels

The NLI model can classify evidence as supportive or contradictory even when the evidence is insufficient to establish the claim.

This is particularly visible in the low NEI recall.

---

### Difficulty with comparative evidence

Some scientific claims depend on comparisons, percentages, experimental conditions, or relationships between multiple pieces of information.

For example:

```text
Treatment A = 69%
Control     = 22%
```

Correct verification requires understanding the relationship between the values rather than simply matching individual sentences.

LLMs can sometimes interpret these relationships differently from the sentence-level NLI model.

---

### LLM and NLI disagreement

The two verification approaches can reach different conclusions from the same evidence.

This allows the benchmark to study complementary failure modes between:

```text
NLI-based verification
        vs
LLM-based verification
```

---

### Conservative fusion errors

The agreement rule intentionally refuses to make a directional decision when the models disagree.

For example:

```text
Gold = SUPPORTED

NLI = SUPPORTED
LLM = NOT ENOUGH EVIDENCE

Hybrid = NOT ENOUGH EVIDENCE
```

If either model was correct, the hybrid still produces an incorrect final prediction.

This explains why the agreement-based hybrid does not necessarily improve over the stronger standalone LLM.

---

# 25. Reproducibility

The following components are frozen before final evaluation:

```text
Dataset construction
        ↓
Retrieval method
        ↓
Semantic/BM25 weights
        ↓
Top-K
        ↓
Retrieved evidence
        ↓
NLI model
        ↓
NLI aggregation
        ↓
NLI threshold
        ↓
LLM prompt
        ↓
LLM temperature
```

The final benchmark is evaluated without using its labels to tune the system.

This provides a controlled comparison between the NLI and LLM verification approaches.

---

# 26. Running the System

For running the paper-processing pipeline, set the PDF path and claims in `main.py`:

```python
PDF_PATH = r"path/to/research_paper.pdf"

CLAIMS = [
    "Your first claim.",
    "Your second claim.",
    "Your third claim."
]
```

Start Ollama:

```bash
ollama serve
```

Then run:

```bash
python main.py
```

---

# 27. Running the Experimental Evaluation

The experimental framework is located inside:

```text
Test/
```

Generate the retrieval cache using:

```bash
python Test/generate_retrieval_cache.py
```

Run the classification experiments using:

```bash
python Test/run_classification.py
```

The individual LLM verification implementations are located in:

```text
Test/llama/
Test/qwen/
Test/mistral/
```

Results are stored in:

```text
Test/results/
```

---

# 28. Technologies Used

### PDF Processing

- PyMuPDF

### Retrieval

- Sentence Transformers
- `all-MiniLM-L6-v2`
- BM25
- `rank_bm25`

### NLI

- Hugging Face Transformers
- DeBERTa-v3-base-mnli-fever-anli
- PyTorch

### LLM Verification

- Ollama
- Llama 3.1 8B
- Qwen3 8B
- Mistral 7B

### Evaluation

- Precision
- Recall
- F1
- Macro-F1
- Accuracy
- Confusion Matrix
- Recall@K
- MRR

---

# 29. Summary

The project implements a complete scientific claim verification pipeline:

```text
Research Paper
      ↓
PDF Processing
      ↓
Text Cleaning
      ↓
Chunking
      ↓
Hybrid Retrieval
      ↓
Top-5 Evidence
      ↓
 ┌───────────────┬────────────────┐
 │               │                │
 ▼               ▼                ▼
DeBERTa        Llama            Qwen/Mistral
 NLI             LLM               LLM
 │               │                │
 └───────────────┴────────────────┘
                 ↓
          Verification Results
                 ↓
          Model Comparison
                 ↓
           Error Analysis
```

The experimental framework evaluates both NLI-based verification and modern local LLM-based verification under the same frozen retrieval conditions.

The final benchmark contains **2,035 claims** across three verification classes, enabling systematic comparison of NLI, LLM, and hybrid verification strategies.
