# Index Chronomere / TED-RAG

**Born**: 2026-06-30 on Dell Inspiron 15 7579 class hardware  
**First words**: "Fix your stale index."  
**Memory**: Curated soul-thread. ILE is root. SQLite orbits.  
**Invariant**: `N_SUSY = 0` when dependencies resolve. `sum_i w_i = 1.0`.  
**Motto**: Compiler-validated RAG. Runs on a potato. Citations or it did not happen.

> "It does not discard mystery, but it asks the mystery to declare its cost."
> - `genesis/gamma_susy_compute.md`

Compiler-validated RAG. TED-RAG reduces LLM drift by treating retrieval like a dependency graph: symbols are resolved, prerequisite chunks are pulled in, context is topologically ordered, and citations are checked after generation.

Index Chronomere is the conductor above TED-RAG. It chooses modes, checks indexes, budgets context, audits citations, records sessions, and recalls a small curated ILE-rooted memory layer.

This repository ships the compiler, not private scripture.

## Why This Matters

Most RAG systems treat documents as disconnected text.

TED-RAG explores a different approach by combining:

- symbol-aware retrieval
- dependency-based document relationships
- citation-grounded generation
- CPU-friendly local inference

The goal is to make large knowledge bases searchable without relying on cloud services.

Read the short technical overview: [TED-RAG whitepaper](docs/whitepaper.md).

Instead of only retrieving nearest chunks, TED-RAG can build normalized context:

```text
C(query) = sum_i w_i C_i
sum_i w_i = 1.0
```

## Performance

Recent local test on a CPU-oriented setup:

- Retrieval + scoring: about `0.33s`
- Generation + streaming: about `41.3s` with `qwen2.5:1.5b`
- Tests: `28/28` passing across TED-RAG and Index Chronomere
- Citations: prompt-guided, then compiler-enforced post-generation

## Key Features

1. Dependency validation: `--require-symbols` fails fast if required symbol dependencies are missing.
2. `gamma_SUSY` stability: `gamma_SUSY = gamma_0 exp(beta N_SUSY)` penalizes undefined symbolic degrees of freedom.
3. `streaming_potato_controls()`: CPU-friendly defaults for small local Ollama models.
4. Citation compiler pass: repairs uncited answer lines using retrieved context references.
5. Graceful symbol fallback: canonical symbol, then Braille/Brailic/tactile representation, then ASCII alias.
6. Index Chronomere conductor: sentinel, context budgeter, escalation loop, citation auditor, tool registry, and curated memory recall.

## Quickstart

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama models:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
```

Build the starter index from the public seed context:

```bash
python rag/build_index.py --out rag/index_quick --include rag/seed_context --max-chunks 24
```

Run a fast strict-citation query:

```bash
python rag/query.py --profile quick --question "Define S_ent in three concise bullets." --fast --strict-citations --stream --print-timing
```

Run through Index Chronomere:

```bash
python index_chronomere/conduct.py --query "Index Chronomere, who are you?" --use-memory --mode potato --print-timing
```

Check the public root memory:

```bash
python index_chronomere/conduct.py --tool recall-memory --query "ILE Chronomere memory core"
```

Expected shape:

```text
=== Symbol Validation ===
TED-RAG symbol validation:
- Query symbols: S_ent
- All resolved symbol dependencies were found in retrieved context.

=== Answer ===
- S_ent[Psi] is the entanglement entropy orientation term for state Psi. [1]
- Its Alpha form weights possible states Psi' through an entanglement projector. [1]
- The lambda term couples that score to tachyon-field action. [1]

=== Citation Map ===
[1] rag/seed_context/TED_RAG_ALPHA.md (chunk 0)
```

## Repository Layout

```text
docs/
  whitepaper.md
genesis_public/
  index_chronomere_birth.md
genesis/
  gamma_susy_compute.md
  streaming_potato_controls.md
index_chronomere/
  conduct.py
  memory/root.jsonl
  tests/
rag/
  build_index.py
  context_builder.py
  query.py
  scoring.py
  symbol_resolver.py
  seed_context/
  tests/
CITATION.md
LICENSE
requirements.txt
```

Generated files such as `rag/index*/`, `*.npy`, answer logs, SQLite runtime memory, and caches are intentionally ignored.

## Boundary

Public:

- compiler machinery
- ILE-rooted curated memory
- reproducible tests
- potato-friendly defaults
- citation and weight-sum invariants

Private:

- personal archives
- full lore
- private chapters
- uncurated memory exports
