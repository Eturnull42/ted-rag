# TED-RAG: Compiler-Validated Retrieval-Augmented Generation

Version: 0.1

## Abstract

TED-RAG is a local Retrieval-Augmented Generation system that treats retrieval as a compiler pipeline instead of a plain nearest-neighbor lookup. It combines symbolic dependency validation, graph-aware context expansion, stability reranking, weighted context construction, streaming local inference, and citation post-processing.

The goal is practical: make large personal or research knowledge bases searchable with lightweight local models while reducing silent hallucination, missing definitions, and uncited claims.

## Problem

Most RAG systems split documents into chunks, embed those chunks, and retrieve the nearest matches for a query. This works well for broad semantic search, but it can fail when the answer depends on symbolic relationships.

Common failure modes include:

- A retrieved chunk uses a symbol without retrieving the definition.
- Related documents are semantically nearby but dependency order is missing.
- The model answers fluently without citations.
- Small local models drift from required notation.
- Context grows too large for CPU-friendly inference.

TED-RAG explores a stricter approach: retrieve context as a dependency-checked, weighted construction.

## Core Claim

TED-RAG treats RAG as a compiler-style pipeline:

```text
query
-> symbol extraction
-> symbol table lookup
-> dependency validation
-> TED expansion
-> Alpha scoring
-> eta/gamma_SUSY stability rerank
-> fractal context builder
-> local LLM generation
-> citation compiler pass
```

The system does not ask the model to remember the rules. It encodes the rules before and after generation.

## Architecture

### Symbol Table

`rag/symbol_table.json` acts like a compiler frontend. Each symbol can define:

- canonical form
- aliases
- dependency list
- source anchors
- formula
- fallback representations

Example symbols include `S_ent`, `Ψ`, `λ`, `Û(t,t')`, `η`, and `γ_SUSY`.

If `--require-symbols` is enabled, unresolved dependencies fail before generation.

### TED Expansion

Temporal Entanglement Dynamics expansion retrieves definitions for symbols used by the query or prior context. This makes dependency retrieval explicit rather than accidental.

The ordering rule is simple: definitions should appear before dependent claims when possible.

### Alpha Scoring

Alpha scoring is an experimental reranker inspired by the Alpha Equation:

```text
S_ent[Ψ] = S_ent[Ψ, ∫ D[Ψ'] |Ψ'⟩⟨Ψ'| e^(-S_ent[Ψ'])] + λ ∫ D[T] S_tachyon[T, Ψ]
```

In implementation, this becomes a retrieval score that blends base similarity, corpus entanglement, and a risk/novelty term.

### Stability Rerank

TED-RAG computes an operational stability adjustment:

```text
η = η_0 (D_f - 1)^α
γ_SUSY = γ_0 exp(β N_SUSY)
score' = score * η / γ_SUSY
```

Here `N_SUSY` is not a physics measurement. It is an implementation metric: the count of unresolved symbol-like tokens in a chunk. More unresolved symbols create a larger penalty.

### Fractal Context Builder

The final context is represented as a normalized weighted construction:

```text
C(query) = Σ w_i C_i
Σ w_i = 1.0
```

This gives the system an auditable context budget. The model receives selected chunks plus their normalized roles and weights.

### Citation Compiler Pass

Strict citation mode uses two layers:

1. Prompt rules requiring inline citations.
2. A post-generation pass that repairs uncited answer lines using retrieved context references.

This is deliberately compiler-like: citation failure becomes detectable and repairable instead of silent.

## Local Inference

`streaming_potato_controls()` provides CPU-friendly defaults:

- `qwen2.5:1.5b`
- `top_k = 2`
- `max_chunk_chars = 600`
- `num_predict = 120`
- `temperature = 0`
- Alpha scoring enabled
- stability rerank enabled
- fractal context enabled

This keeps the pipeline usable on ordinary local hardware.

## Evaluation

The current public demo uses the following checks:

- Unit tests: `8/8` passing.
- Symbol validation: query symbols resolve through `symbol_table.json`.
- Dependency coverage: required dependencies are found in retrieved context.
- Context normalization: weight sum reports `1.000000`.
- Citation presence: generated claims should include `[1]`, `[2]`, etc.
- Citation repair: uncited answer lines are repaired by the citation pass.
- Runtime timing: retrieval and generation timings are reported separately.
- Export safety: generated indexes, embeddings, caches, and private corpus chunks are not shipped.

Recent local timing:

```text
retrieval + scoring: about 0.33s
generation + streaming: about 41.3s with qwen2.5:1.5b
tests: 8/8 in about 0.098s
```

## Reproducible Demo

Build a small public seed index:

```bash
python rag/build_index.py --out rag/index_quick --include rag/seed_context --max-chunks 24
```

Run:

```bash
python rag/query.py --profile quick --question "Define S_ent in three concise bullets." --fast --strict-citations --stream --print-timing
```

Expected output shape:

```text
=== Symbol Validation ===
TED-RAG symbol validation:
- Query symbols: S_ent
- All resolved symbol dependencies were found in retrieved context.

=== Fractal Constraint ===
Fractal Constraint:
- Formula: C(query) = Σ w_i C_i
- Weight sum: 1.000000

=== Answer ===
- S_ent[Ψ] is the entanglement entropy orientation term for state Ψ. [1]
- Its Alpha form weights possible states Ψ' through an entanglement projector. [1]
- The λ term couples that score to tachyon-field action. [1]
```

## Boundaries

TED-RAG is experimental retrieval infrastructure. The physics-inspired names and equations guide system design, scoring, and uncertainty discipline, but this repository does not claim to prove a physical theory.

The public repository ships a small seed context and example symbol table. Users should rebuild indexes from their own corpora and validate outputs against their own source documents.

## Summary

TED-RAG asks a local model to answer only after the retrieval system has done compiler-like work:

- define symbols
- resolve dependencies
- order context
- weight evidence
- constrain generation
- check citations

The result is a local-first RAG pattern for symbolic archives where definitions, dependencies, and citations matter.
