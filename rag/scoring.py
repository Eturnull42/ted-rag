#!/usr/bin/env python3
"""Experimental Alpha/TED-aware scoring helpers for local RAG retrieval."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np

from symbol_resolver import extract_symbols


@dataclass(frozen=True)
class AlphaScore:
    total: float
    base: float
    entanglement: float
    tachyon_risk: float
    tachyon_novelty: float
    lambda_risk: float
    lambda_novelty: float


@dataclass(frozen=True)
class StabilityScore:
    adjusted: float
    base_score: float
    eta: float
    gamma_susy: float
    fractal_dimension: float
    unresolved_symbols: int


TACHYON_SYMBOLS = {"T_μν^tachyon", "S_tachyon", "T"}
TACHYON_TERMS = {
    "tachyon",
    "tachyonic",
    "unstable",
    "instability",
    "risk",
    "penalty",
    "collapse",
    "decoherence",
}

SYMBOLISH_PATTERN = re.compile(
    r"""
    (?:[A-Za-z]+_[A-Za-z0-9_]+(?:\[[^\]]+\])?)
    |(?:[A-Za-z]\[[^\]]+\])
    |(?:[^\x00-\x7F])
    """,
    re.VERBOSE,
)


def sample_corpus_embeddings(embeddings: np.ndarray, sample_size: int) -> np.ndarray:
    if sample_size <= 0 or embeddings.shape[0] <= sample_size:
        return embeddings
    indices = np.linspace(0, embeddings.shape[0] - 1, sample_size, dtype=np.int64)
    return embeddings[indices]


def tachyon_risk(chunk_text: str, symbol_table: Dict[str, Any]) -> float:
    lowered = chunk_text.lower()
    term_hits = sum(1 for term in TACHYON_TERMS if term in lowered)

    symbol_hits = 0
    if symbol_table:
        symbols = {match.symbol for match in extract_symbols(chunk_text, symbol_table)}
        symbol_hits = len(symbols.intersection(TACHYON_SYMBOLS))

    raw = term_hits + symbol_hits
    return min(1.0, raw / 4.0)


def known_symbol_terms(symbol_table: Dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    symbols = symbol_table.get("symbols", {})
    if not isinstance(symbols, dict):
        return terms

    for symbol, entry in symbols.items():
        terms.add(str(symbol))
        if not isinstance(entry, dict):
            continue
        aliases = entry.get("aliases", [])
        if isinstance(aliases, list):
            for alias in aliases:
                if isinstance(alias, str):
                    terms.add(alias)
    return terms


def estimate_fractal_dimension(chunk_text: str) -> float:
    lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
    if not lines:
        return 1.4

    equation_lines = sum(1 for line in lines if any(mark in line for mark in ("=", "∫", "Σ", "∂", "∇", "∞", "^")))
    table_lines = sum(1 for line in lines if line.count("|") >= 2)
    code_lines = sum(
        1
        for line in lines
        if line.startswith(("def ", "class ", "for ", "if ", "return ", "import "))
        or line.endswith(":")
        or "```" in line
    )
    symbol_chars = sum(1 for char in chunk_text if not char.isalnum() and not char.isspace())
    tokens = re.findall(r"\w+", chunk_text, flags=re.UNICODE)
    unique_ratio = len(set(tokens)) / max(1, len(tokens))

    structure = (equation_lines + table_lines + code_lines) / max(1, len(lines))
    symbol_density = symbol_chars / max(1, len(chunk_text))
    raw = 1.45 + (0.28 * min(1.0, structure * 2.5)) + (0.15 * min(1.0, symbol_density * 8.0))
    raw += 0.08 * min(1.0, unique_ratio)
    return max(1.35, min(1.92, raw))


def unresolved_symbol_count(chunk_text: str, symbol_table: Dict[str, Any]) -> int:
    known = known_symbol_terms(symbol_table)
    known_lower = {term.lower() for term in known if term.isascii()}
    resolved = {match.matched for match in extract_symbols(chunk_text, symbol_table)}
    resolved.update(match.symbol for match in extract_symbols(chunk_text, symbol_table))

    unresolved: set[str] = set()
    for match in SYMBOLISH_PATTERN.finditer(chunk_text):
        token = match.group(0).strip()
        if not token or token in resolved or token in known:
            continue
        if token.isascii() and token.lower() in known_lower:
            continue
        if token in {"|", "⟩", "⟨"}:
            continue
        unresolved.add(token)
    return len(unresolved)


def stability_score(
    base_score: float,
    chunk_text: str,
    symbol_table: Dict[str, Any],
    eta0: float = 1.0,
    eta_alpha: float = 1.0,
    gamma0: float = 1.0,
    gamma_beta: float = 0.08,
) -> StabilityScore:
    df = estimate_fractal_dimension(chunk_text)
    unresolved = unresolved_symbol_count(chunk_text, symbol_table)
    eta = eta0 * max(0.01, df - 1.0) ** eta_alpha
    gamma = gamma0 * math.exp(gamma_beta * unresolved)
    adjusted = base_score * eta / gamma
    return StabilityScore(
        adjusted=adjusted,
        base_score=base_score,
        eta=eta,
        gamma_susy=gamma,
        fractal_dimension=df,
        unresolved_symbols=unresolved,
    )


def stability_rerank(
    ranked: List[tuple[float, Dict[str, object]]],
    symbol_table: Dict[str, Any],
    eta0: float = 1.0,
    eta_alpha: float = 1.0,
    gamma0: float = 1.0,
    gamma_beta: float = 0.08,
    preserve_ted_definitions: bool = True,
) -> List[tuple[float, Dict[str, object]]]:
    scored: List[tuple[float, Dict[str, object]]] = []
    for score, chunk in ranked:
        updated = dict(chunk)
        stability = stability_score(
            base_score=score,
            chunk_text=str(updated.get("text", "")),
            symbol_table=symbol_table,
            eta0=eta0,
            eta_alpha=eta_alpha,
            gamma0=gamma0,
            gamma_beta=gamma_beta,
        )
        updated["stability_score"] = {
            "adjusted": stability.adjusted,
            "base_score": stability.base_score,
            "eta": stability.eta,
            "gamma_susy": stability.gamma_susy,
            "fractal_dimension": stability.fractal_dimension,
            "unresolved_symbols": stability.unresolved_symbols,
        }
        scored.append((stability.adjusted, updated))

    scored.sort(
        key=lambda item: (
            0 if preserve_ted_definitions and item[1].get("ted_role") == "definition" else 1,
            -float(item[0]),
            str(item[1].get("source", "")),
            int(item[1].get("chunk_id", -1)),
        )
    )
    return scored


def alpha_score(
    query_vec: np.ndarray,
    chunk_vec: np.ndarray,
    corpus_embeddings: np.ndarray,
    chunk_text: str,
    symbol_table: Dict[str, Any],
    lambda_risk: float = 0.1,
    lambda_novelty: float = 0.0,
) -> AlphaScore:
    base = float(np.dot(query_vec, chunk_vec))

    if corpus_embeddings.size == 0:
        entanglement = 0.0
    else:
        chunk_to_corpus = corpus_embeddings @ chunk_vec
        query_to_corpus = corpus_embeddings @ query_vec
        weights = np.exp(-chunk_to_corpus)
        entanglement = float(np.mean(weights * query_to_corpus))

    risk = tachyon_risk(chunk_text, symbol_table)
    novelty = risk
    total = base + entanglement + (lambda_novelty * novelty) - (lambda_risk * risk)
    return AlphaScore(
        total=total,
        base=base,
        entanglement=entanglement,
        tachyon_risk=risk,
        tachyon_novelty=novelty,
        lambda_risk=lambda_risk,
        lambda_novelty=lambda_novelty,
    )


def alpha_scores_for_candidates(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    chunks: List[Dict[str, object]],
    candidate_indices: Iterable[int],
    symbol_table: Dict[str, Any],
    lambda_risk: float = 0.1,
    lambda_novelty: float = 0.0,
    entanglement_sample: int = 512,
) -> List[tuple[int, AlphaScore]]:
    corpus_sample = sample_corpus_embeddings(embeddings, entanglement_sample)
    scored: List[tuple[int, AlphaScore]] = []

    for idx in candidate_indices:
        chunk = chunks[int(idx)]
        score = alpha_score(
            query_vec=query_vec,
            chunk_vec=embeddings[int(idx)],
            corpus_embeddings=corpus_sample,
            chunk_text=str(chunk.get("text", "")),
            symbol_table=symbol_table,
            lambda_risk=lambda_risk,
            lambda_novelty=lambda_novelty,
        )
        scored.append((int(idx), score))

    scored.sort(key=lambda item: item[1].total, reverse=True)
    return scored
