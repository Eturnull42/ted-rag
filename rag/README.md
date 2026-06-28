# Enigma Stone TED-RAG (Local, Ollama)

This setup builds a local retrieval index over your Enigma Stone corpus and answers questions with source citations. The TED-RAG layer adds canonical seed context for Temporal Entanglement Dynamics, EQG/tachyon action terms, the Alpha Equation, and the Relic/Brailic symbolic language before retrieving from the wider notes.

## Why this stack

- Generation model: `qwen2.5:3b-instruct-q4_K_M` (good math/code reasoning for small RAM).
- Embeddings model: `nomic-embed-text` (lightweight and solid retrieval quality).
- Storage: local `npy` + `jsonl` files (fast, simple, no DB server).
- Seed context: `rag/seed_context` is indexed first and gets a small query-time priority boost for TED/EQG/tachyon/fractal/infinity-symbol and Relic/Brailic glyph-language questions.

## 1) Install prerequisites

1. Install Ollama: https://ollama.com/download
2. Pull models:

```powershell
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull nomic-embed-text
```

3. In this folder, install Python deps:

```powershell
py -3 -m pip install -r rag/requirements.txt
```

## 2) Build the index

```powershell
py -3 rag/build_index.py --root "." --out "rag/index"
```

Optional knobs:

- `--chunk-size 1200`
- `--chunk-overlap 180`
- `--embed-model nomic-embed-text`
- `--include "rag/seed_context,book,binary,Binary Enigma,notes/markdown,zip_imports"`
- `--exclude-ext ".jpeg,.jpg,.png,.pdf,.bin,.pkl"`

By default, the indexer excludes these extensions to keep retrieval text-only:

- `.jpeg`
- `.jpg`
- `.png`
- `.pdf`
- `.bin`
- `.pkl`

## 3) Ask questions

```powershell
py -3 rag/query.py --question "Derive the key temporal equation and explain assumptions."
```

Useful options:

- `--top-k 8`
- `--llm-model qwen2.5:3b-instruct-q4_K_M`
- `--embed-model nomic-embed-text`
- `--show-context`
- `--stream`
- `--print-timing`
- `--strict-citations`
- `--fast`
- `--num-predict 512`
- `--temperature 0.2`
- `--stop "[END]"`
- `--generation-timeout 240`
- `--max-chunk-chars 800`
- `--symbol-depth 1`
- `--require-symbols`
- `--ted-expansion-limit 8`
- `--no-ted-expand`
- `--alpha-score`
- `--alpha-candidates 64`
- `--alpha-entanglement-sample 512`
- `--alpha-lambda-risk 0.1`
- `--alpha-lambda-novelty 0.0`
- `--stability-rerank`
- `--eta0 1.0`
- `--eta-alpha 1.0`
- `--gamma0 1.0`
- `--gamma-beta 0.08`
- `--fractal-context`
- `--constraint-weight-floor 0.001`
- `--constraint-summary-limit 12`

TED-RAG symbol validation:

- By default, `query.py` resolves symbols found in the question against `rag/symbol_table.json`, scans the retrieved context, and prints warnings for dependencies missing from context.
- Use `--require-symbols` to hard-fail before generation when required symbol dependencies are not present in retrieved context.
- Use `--symbol-depth` to control dependency expansion depth during validation.
- If a canonical symbol is missing from retrieved context or cannot be rendered, TED-RAG reports an explicit representation fallback from the symbol table: Braille/Brailic/tactile fields first, then semantic aliases, then ASCII aliases.

TED expansion and context ordering:

- After initial vector retrieval, `query.py` resolves query symbols and pulls indexed chunks from each symbol's `defined_in` sources.
- If a `defined_in` source is not present in the current index, `query.py` reads that source file directly and injects the referenced Markdown section as a TED definition fallback.
- Expanded definition chunks are ordered before ordinary retrieved chunks.
- Definitions are sorted by dependency depth so lower-level dependencies appear before the symbols that depend on them.
- Use `--ted-expansion-limit` to control how many definition chunks are added.
- Use `--no-ted-expand` to inspect the raw vector retrieval path without TED expansion.

Experimental Alpha scoring:

- Use `--alpha-score` to rerank the initial vector candidate pool with `rag/scoring.py`.
- The experimental score combines base query similarity, an average corpus-entanglement term, a tachyonic-risk penalty, and an optional tachyonic-novelty bonus.
- `--alpha-candidates` controls how many vector-nearest candidates are reranked.
- `--alpha-entanglement-sample` controls how many corpus embeddings are sampled for the entanglement term.
- `--alpha-lambda-risk` tunes the tachyonic-risk penalty weight.
- `--alpha-lambda-novelty` tunes the tachyonic-novelty exploration bonus.
- Alpha scoring is opt-in; without `--alpha-score`, retrieval uses the existing vector plus lexical/source-priority path.

Experimental error correction/decoherence reranking:

- Use `--stability-rerank` to apply `score = score * eta / gamma_SUSY` after retrieval and TED expansion.
- `eta = eta0 * (D_f - 1)^eta_alpha`, where `D_f` is estimated from equation density, table/code structure, symbol density, and token variety.
- `gamma_SUSY = gamma0 * exp(gamma_beta * N_SUSY)`, where `N_SUSY` is an operational count of unresolved symbol-like tokens in the chunk.
- TED definition chunks are preserved before ordinary chunks, then reranked within their groups.
- With `--show-context`, stability diagnostics print as `eta`, `gamma`, `D_f`, and `N_SUSY`.

Experimental fractal constraint context:

- Use `--fractal-context` to turn the final ranked context into weighted terms: `C(query) = sum_i w_i C_i`.
- Each context item receives `constraint_term`, `constraint_weight`, and `constraint_role` metadata.
- Weights are normalized from the final retrieval/reranking scores with a small positive floor.
- The prompt receives a compact Fractal Constraint summary, while citations still point to the underlying context items.
- Use `--constraint-summary-limit` to control how many weighted terms appear in the prompt summary.

## 4) TED-RAG compiler pipeline

The full opt-in pipeline is:

```text
query
-> vector retrieval
-> symbol resolution
-> dependency validation
-> TED definition expansion
-> optional Alpha scoring
-> optional eta/gamma_SUSY stability reranking
-> optional fractal constraint context build
-> prompt with citations
```

Recommended experimental full-stack query:

```powershell
py -3 rag/query.py --profile quick --question "Unify S_ent and TED assumptions." --alpha-score --stability-rerank --fractal-context --show-context
```

Keep the default path for normal work. Use the full stack when you want to inspect how the compiler-style retrieval layers are shaping context.

Streaming generation:

- Use `--stream` for long local Ollama generations so tokens print as they arrive.
- Use `--num-predict` to cap answer length for heavyweight prompts.
- Use `--stop` to pass optional Ollama stop sequences; repeat the flag for multiple stops.
- Use `--generation-timeout` only for non-streaming generation.
- Use `--max-chunk-chars` to cap each retrieved context chunk before it enters the prompt.
- Use `--print-timing` to print retrieval, prompt, and generation timing diagnostics.
- Use `--strict-citations` to add citation-specific prompt rules and run a post-generation citation compiler pass.
- Use `--fast` to apply `streaming_potato_controls()`: qwen2.5:1.5b, top-k 2, Alpha scoring, stability rerank, fractal context, max chunk chars 600, num_predict 120, temperature 0.

Example:

```powershell
py -3 rag/query.py --profile quick --question "Define S_ent in five bullets." --stream --num-predict 256
```

Strict citation fast test:

```powershell
py -3 rag/query.py --profile quick --question "Define S_ent in three concise bullets." --fast --strict-citations --stream --print-timing
```

Potato-friendly full stack:

```powershell
ollama pull qwen2.5:1.5b
py -3 rag/query.py --profile quick --question "Define S_ent in five bullets." --llm-model qwen2.5:1.5b --top-k 3 --alpha-score --alpha-candidates 24 --stability-rerank --fractal-context --stream --num-predict 256 --max-chunk-chars 800 --print-timing
```

Performance notes:

- `gamma_SUSY` runs after retrieval/TED expansion on contenders, not over the whole corpus.
- `--alpha-candidates` is the top-K prefilter before Alpha scoring.
- Fractal Constraint summaries print before generation, so long local generations show useful progress early.
- If Ollama generation dominates timing, try `qwen2.5:1.5b`, lower `--num-predict`, lower `--top-k`, or smaller `--max-chunk-chars`.

Profile shortcuts:

- Quick profile (100 chunks):

```powershell
py -3 rag/query.py --profile quick --question "Summarize temporal operators."
```

- Medium profile (500 chunks):

```powershell
py -3 rag/query.py --profile medium --question "Compare temporal entanglement equation variants."
```

- Deep profile (4000 chunks):

```powershell
py -3 rag/query.py --profile deep --question "Unify temporal entanglement equations and list assumptions."
```

## Tests

Run the offline TED-RAG tests with:

```powershell
py -3 -m unittest discover -s rag/tests
```

These tests cover symbol resolution, missing dependency validation, Alpha risk/novelty scoring, stability reranking, and fractal constraint weights. They do not call Ollama.

## Example math-heavy prompts

- "Unify the symbol system around \u221e, \u2194, and temporal entanglement into a minimal algebra."
- "Extract every equation-like statement, normalize variables, and list contradictions."
- "Find all recurrence relations in the corpus and propose closed forms where plausible."
- "Map symbol definitions to code-like operators and produce a typed grammar."
- "Use the TED-RAG Alpha context to unify TED, EQG tachyon dynamics, and fractal constraint handling."
- "Use the Relic/Brailic language context to compress this equation into a glyph table entry."

## Notes

- Rebuild index after major corpus changes.
- Rebuild all active profiles after editing `rag/seed_context`.
- The answer output includes citations in `[n]` format mapped to source files.
- If Ollama is not running, start it and re-run the command.
