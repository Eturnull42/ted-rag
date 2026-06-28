#!/usr/bin/env python3
"""Fractal constraint context builder for TED-RAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ConstraintTerm:
    index: int
    weight: float
    raw_score: float
    source: str
    chunk_id: int
    role: str
    symbols: list[str]


def normalized_weights(scores: list[float], floor: float = 0.001) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    shifted = [score - min_score + floor for score in scores]
    total = sum(shifted)
    if total <= 0.0:
        return [1.0 / len(scores) for _ in scores]
    return [value / total for value in shifted]


def build_fractal_constraint(
    ranked: List[tuple[float, Dict[str, object]]],
    weight_floor: float = 0.001,
) -> tuple[List[tuple[float, Dict[str, object]]], dict[str, object]]:
    weights = normalized_weights([float(score) for score, _ in ranked], floor=weight_floor)
    terms: list[ConstraintTerm] = []
    weighted_ranked: List[tuple[float, Dict[str, object]]] = []

    for idx, ((score, chunk), weight) in enumerate(zip(ranked, weights, strict=False), start=1):
        updated = dict(chunk)
        role = str(updated.get("ted_role", "retrieved"))
        symbols = [str(symbol) for symbol in updated.get("ted_symbols", [])]
        term = ConstraintTerm(
            index=idx,
            weight=weight,
            raw_score=float(score),
            source=str(updated.get("source", "")),
            chunk_id=int(updated.get("chunk_id", -1)),
            role=role,
            symbols=symbols,
        )
        updated["constraint"] = {
            "term": f"C_{idx}",
            "weight": term.weight,
            "raw_score": term.raw_score,
            "role": term.role,
            "symbols": term.symbols,
        }
        terms.append(term)
        weighted_ranked.append((score, updated))

    summary = {
        "formula": "C(query) = Σ w_i C_i",
        "weight_sum": sum(weights),
        "terms": [
            {
                "term": f"C_{term.index}",
                "weight": term.weight,
                "raw_score": term.raw_score,
                "source": term.source,
                "chunk_id": term.chunk_id,
                "role": term.role,
                "symbols": term.symbols,
            }
            for term in terms
        ],
    }
    return weighted_ranked, summary


def format_constraint_summary(summary: dict[str, object], max_terms: int | None = None) -> str:
    terms = summary.get("terms", [])
    if not isinstance(terms, list) or not terms:
        return "Fractal Constraint: not constructed."

    limit = len(terms) if max_terms is None else max(0, min(max_terms, len(terms)))
    lines = [
        "Fractal Constraint:",
        f"- Formula: {summary.get('formula', 'C(query) = Σ w_i C_i')}",
        f"- Weight sum: {float(summary.get('weight_sum', 0.0)):.6f}",
    ]
    for term in terms[:limit]:
        if not isinstance(term, dict):
            continue
        symbols = ", ".join(str(symbol) for symbol in term.get("symbols", [])) or "(none)"
        lines.append(
            "- "
            f"{term.get('term')}: "
            f"w={float(term.get('weight', 0.0)):.6f}, "
            f"role={term.get('role')}, "
            f"source={term.get('source')}#{term.get('chunk_id')}, "
            f"symbols={symbols}"
        )

    omitted = len(terms) - limit
    if omitted > 0:
        lines.append(f"- Omitted terms: {omitted}")
    return "\n".join(lines)
