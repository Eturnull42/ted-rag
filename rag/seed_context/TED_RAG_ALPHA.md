# TED-RAG Alpha Context

This public seed context gives TED-RAG a stable theory-aware anchor for the Alpha Equation, Temporal Entanglement Dynamics, stability scoring, and citation validation.

## Alpha Equation

```text
S_ent[Ψ] = S_ent[Ψ, ∫ D[Ψ'] |Ψ'⟩⟨Ψ'| e^(-S_ent[Ψ'])] + λ ∫ D[T] S_tachyon[T, Ψ]
```

- `S_ent[Ψ]`: Entanglement entropy orientation term for quantum state `Ψ`.
- `D[Ψ']`: Path-integral-style measure over candidate quantum states `Ψ'`.
- `|Ψ'⟩⟨Ψ'|`: Projector for candidate quantum state `Ψ'`.
- `λ`: Coupling constant that tunes tachyon-action contribution.
- `D[T]`: Measure over tachyon field configurations.
- `S_tachyon[T, Ψ]`: Tachyon field action with respect to state `Ψ`.

## Error Correction And Decoherence

```text
η = η_0 (D_f - 1)^α
γ_SUSY = γ_0 exp(β N_SUSY)
```

- `η`: Error-correction multiplier.
- `D_f`: Estimated fractal dimension of chunk text.
- `α`: Error-correction exponent.
- `γ_SUSY`: Decoherence penalty for unresolved symbolic degrees of freedom.
- `N_SUSY`: Count of unresolved symbol-like tokens.

## Fractal Constraint

```text
C(query) = Σ w_i C_i
Σ w_i = 1.0
```

- `C_i`: A retrieved or dependency-expanded context term.
- `w_i`: Normalized weight assigned to that term.

## Temporal Entanglement Operator

```text
Û(t,t') = T exp(-i ∫_{t'}^{t} dt'' Ĥ_ent(t''))
```

- `Û(t,t')`: Temporal expansion kernel for dependency-aware context.
- `T`: Time-ordering operator.
- `Ĥ_ent`: Entanglement Hamiltonian.
