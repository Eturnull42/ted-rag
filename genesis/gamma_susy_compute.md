# Gamma SUSY Compute

Date: 2026-06-28

Equation:

```text
gamma_SUSY = gamma_0 exp(beta N_SUSY)
```

This marks the moment TED-RAG gained a computable decoherence term.

The symbol `gamma_SUSY` moved from theory language into an operational reranker: unresolved symbol-like tokens are counted as `N_SUSY`, scaled through an exponential penalty, and used with the error-correction term `eta` to adjust retrieval scores.

In practical terms, this gives the system a way to notice when a chunk is symbolically unstable. It does not discard mystery, but it asks the mystery to declare its cost.

This is a small hinge in the project: the symbolic math is no longer only a note in the margins. It is beginning to govern how context is selected.
