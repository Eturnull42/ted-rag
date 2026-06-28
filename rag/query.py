#!/usr/bin/env python3
"""Query a local TED-RAG embedding index and answer with Ollama."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests

from context_builder import build_fractal_constraint, format_constraint_summary
from scoring import alpha_scores_for_candidates, stability_rerank
from symbol_resolver import (
    extract_symbols,
    fallback_representations,
    load_symbol_table,
    resolve_dependencies,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query TED-RAG index")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--index", default="rag/index", help="Index directory")
    parser.add_argument(
        "--profile",
        choices=["default", "quick", "medium", "deep"],
        default="default",
        help="Index profile: default=--index path, quick=rag/index_quick, medium=rag/index_medium, deep=rag/index_deep",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Number of retrieved chunks")
    parser.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model")
    parser.add_argument("--llm-model", default="qwen2.5:3b-instruct-q4_K_M", help="Ollama generation model")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved context before answer")
    parser.add_argument("--stream", action="store_true", help="Stream Ollama generation tokens as they arrive")
    parser.add_argument("--print-timing", action="store_true", help="Print retrieval and generation timing diagnostics")
    parser.add_argument("--strict-citations", action="store_true", help="Require and post-process inline citations")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use potato-friendly defaults: qwen2.5:1.5b, top-k 2, max chunk chars 600, num_predict 120, temperature 0",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help="Optional Ollama num_predict generation limit",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional Ollama generation temperature",
    )
    parser.add_argument(
        "--stop",
        action="append",
        default=[],
        help="Optional Ollama stop sequence; repeat for multiple stops",
    )
    parser.add_argument(
        "--generation-timeout",
        type=int,
        default=240,
        help="Read timeout in seconds for non-streaming Ollama generation",
    )
    parser.add_argument("--symbol-table", default="rag/symbol_table.json", help="TED-RAG symbol table path")
    parser.add_argument(
        "--symbol-depth",
        type=int,
        default=1,
        help="Dependency depth for query-time symbol validation",
    )
    parser.add_argument(
        "--require-symbols",
        action="store_true",
        help="Fail before generation if resolved symbol dependencies are missing from retrieved context",
    )
    parser.add_argument(
        "--no-ted-expand",
        action="store_true",
        help="Disable TED dependency expansion and topological context ordering",
    )
    parser.add_argument(
        "--ted-expansion-limit",
        type=int,
        default=8,
        help="Maximum definition chunks to add from resolved TED-RAG dependencies",
    )
    parser.add_argument(
        "--alpha-score",
        action="store_true",
        help="Experimentally rerank retrieval candidates with Alpha/TED-aware scoring",
    )
    parser.add_argument(
        "--alpha-candidates",
        type=int,
        default=64,
        help="Candidate pool size for --alpha-score reranking",
    )
    parser.add_argument(
        "--alpha-entanglement-sample",
        type=int,
        default=512,
        help="Corpus sample size used for the Alpha entanglement term",
    )
    parser.add_argument(
        "--alpha-lambda-risk",
        type=float,
        default=0.1,
        help="Risk penalty weight for tachyonic/unstable chunks during --alpha-score",
    )
    parser.add_argument(
        "--alpha-lambda-novelty",
        type=float,
        default=0.0,
        help="Novelty bonus weight for tachyonic/unstable chunks during --alpha-score",
    )
    parser.add_argument(
        "--stability-rerank",
        action="store_true",
        help="Experimentally rerank final context with eta/gamma_SUSY error-correction and decoherence weights",
    )
    parser.add_argument("--eta0", type=float, default=1.0, help="Base eta weight for --stability-rerank")
    parser.add_argument(
        "--eta-alpha",
        type=float,
        default=1.0,
        help="Exponent alpha in eta = eta0 * (D_f - 1)^alpha",
    )
    parser.add_argument("--gamma0", type=float, default=1.0, help="Base gamma weight for --stability-rerank")
    parser.add_argument(
        "--gamma-beta",
        type=float,
        default=0.08,
        help="Beta in gamma_SUSY = gamma0 * exp(beta * unresolved_symbols)",
    )
    parser.add_argument(
        "--fractal-context",
        action="store_true",
        help="Build final context as weighted fractal constraint terms C(query) = sum w_i C_i",
    )
    parser.add_argument(
        "--constraint-weight-floor",
        type=float,
        default=0.001,
        help="Small positive floor used when normalizing fractal context weights",
    )
    parser.add_argument(
        "--constraint-summary-limit",
        type=int,
        default=12,
        help="Maximum fractal constraint terms to show in the prompt summary",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=None,
        help="Optional maximum characters from each context chunk included in the prompt",
    )
    return parser.parse_args()


def streaming_potato_controls(args: argparse.Namespace) -> argparse.Namespace:
    """Apply CPU-friendly defaults for small local Ollama models."""
    args.llm_model = "qwen2.5:1.5b"
    args.top_k = min(args.top_k, 2)
    args.max_chunk_chars = args.max_chunk_chars or 600
    args.num_predict = args.num_predict or 120
    args.temperature = 0.0 if args.temperature is None else args.temperature
    args.alpha_score = True
    args.alpha_candidates = min(args.alpha_candidates, 24)
    args.stability_rerank = True
    args.fractal_context = True
    args.constraint_summary_limit = min(args.constraint_summary_limit, 2)
    return args


def load_index(index_dir: Path) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    embeddings_path = index_dir / "embeddings.npy"
    chunks_path = index_dir / "chunks.jsonl"
    if not embeddings_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"Missing index files in {index_dir}")

    embeddings = np.load(embeddings_path)
    chunks: List[Dict[str, object]] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            chunks.append(json.loads(line))

    if embeddings.shape[0] != len(chunks):
        raise RuntimeError("Index mismatch: embeddings row count does not match chunk count")

    return embeddings, chunks


def embed_query(base_url: str, model: str, question: str) -> np.ndarray:
    endpoint = f"{base_url.rstrip('/')}/api/embeddings"
    response = requests.post(endpoint, json={"model": model, "prompt": question}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    vec = np.asarray(payload.get("embedding", []), dtype=np.float32)
    if vec.size == 0:
        raise RuntimeError("No embedding returned for question")

    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def lexical_math_score(question: str, chunk_text: str) -> float:
    q_tokens = {t for t in question.lower().split() if len(t) > 2}
    c_tokens = set(chunk_text.lower().split())
    overlap = len(q_tokens.intersection(c_tokens))

    symbol_bonus = 0.0
    for sym in ["=", "+", "-", "*", "/", "^", "=>", "lambda", "\u221e", "\u03a3", "\u2202", "\u2207", "\u03a8", "\u03bb"]:
        if sym in question and sym in chunk_text:
            symbol_bonus += 0.2

    return overlap * 0.05 + symbol_bonus


def source_priority_score(question: str, chunk: Dict[str, object]) -> float:
    source = str(chunk.get("source", "")).replace("\\", "/")
    if not source.startswith("rag/seed_context/"):
        return 0.0

    ted_terms = {
        "ted",
        "temporal",
        "entanglement",
        "eqg",
        "alpha",
        "tachyon",
        "fractal",
        "infinity",
        "symbol",
        "symbolic",
        "glyph",
        "glyphic",
        "compression",
        "compress",
        "relic",
        "tablet",
        "tablets",
        "brailic",
        "braillic",
        "braille",
        "language",
        "lexicon",
        "tactile",
        "somatosensory",
        "mineral",
        "hamiltonian",
        "stress-energy",
    }
    q_tokens = set(question.lower().replace("_", " ").replace("-", " ").split())
    if ted_terms.intersection(q_tokens):
        return 0.35
    return 0.15


def retrieve(
    question: str,
    q_vec: np.ndarray,
    embeddings: np.ndarray,
    chunks: List[Dict[str, object]],
    top_k: int,
    symbol_table: Dict[str, Any] | None = None,
    alpha_enabled: bool = False,
    alpha_candidates: int = 64,
    alpha_entanglement_sample: int = 512,
    alpha_lambda_risk: float = 0.1,
    alpha_lambda_novelty: float = 0.0,
) -> List[Tuple[float, Dict[str, object]]]:
    sim = embeddings @ q_vec
    top_k = max(1, min(top_k, len(chunks)))
    pool_size = top_k
    if alpha_enabled:
        pool_size = max(top_k, min(max(1, alpha_candidates), len(chunks)))
    top_idx = np.argpartition(sim, -pool_size)[-pool_size:]

    scored: List[Tuple[float, Dict[str, object]]] = []
    if alpha_enabled:
        if symbol_table is None:
            raise ValueError("--alpha-score requires a loaded symbol table")
        alpha_ranked = alpha_scores_for_candidates(
            query_vec=q_vec,
            embeddings=embeddings,
            chunks=chunks,
            candidate_indices=top_idx,
            symbol_table=symbol_table,
            lambda_risk=alpha_lambda_risk,
            lambda_novelty=alpha_lambda_novelty,
            entanglement_sample=alpha_entanglement_sample,
        )
        for idx, alpha in alpha_ranked:
            chunk = dict(chunks[int(idx)])
            chunk["alpha_score"] = {
                "total": alpha.total,
                "base": alpha.base,
                "entanglement": alpha.entanglement,
                "tachyon_risk": alpha.tachyon_risk,
                "tachyon_novelty": alpha.tachyon_novelty,
                "lambda_risk": alpha.lambda_risk,
                "lambda_novelty": alpha.lambda_novelty,
            }
            lex = lexical_math_score(question, str(chunk["text"]))
            priority = source_priority_score(question, chunk)
            score = alpha.total + lex + priority
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    for idx in top_idx:
        base = float(sim[idx])
        chunk = chunks[int(idx)]
        lex = lexical_math_score(question, str(chunk["text"]))
        priority = source_priority_score(question, chunk)
        scored.append((base + lex + priority, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def chunk_key(chunk: Dict[str, object]) -> tuple[str, int]:
    return (str(chunk.get("source", "")).replace("\\", "/"), int(chunk.get("chunk_id", -1)))


def anchor_source(anchor: str) -> str:
    return anchor.split("#", 1)[0].replace("\\", "/")


def anchor_fragment(anchor: str) -> str:
    parts = anchor.split("#", 1)
    return parts[1] if len(parts) == 2 else ""


def markdown_slug(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE)
    slug = re.sub(r"[\s_]+", "-", slug.strip())
    return slug


def markdown_section(text: str, fragment: str) -> str:
    if not fragment:
        return text.strip()

    lines = text.splitlines()
    start = None
    start_level = 0
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    for idx, line in enumerate(lines):
        match = heading_pattern.match(line)
        if not match:
            continue
        if markdown_slug(match.group(2)) == fragment:
            start = idx
            start_level = len(match.group(1))
            break

    if start is None:
        return text.strip()

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        match = heading_pattern.match(lines[idx])
        if match and len(match.group(1)) <= start_level:
            end = idx
            break

    return "\n".join(lines[start:end]).strip()


def dependency_depths(resolution: Dict[str, object]) -> Dict[str, int]:
    depths = {str(symbol): 0 for symbol in resolution.get("roots", [])}
    edges = resolution.get("edges", [])
    if not isinstance(edges, list):
        return depths

    for _ in range(max(1, len(edges) + len(depths) + 1)):
        changed = False
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            parent = str(edge.get("from", ""))
            dep = str(edge.get("to", ""))
            next_depth = depths.get(parent, 0) + 1
            if next_depth > depths.get(dep, -1):
                depths[dep] = next_depth
                changed = True
        if not changed:
            break
    return depths


def symbols_defined_by_source(
    resolution: Dict[str, object],
) -> Dict[str, set[str]]:
    by_source: Dict[str, set[str]] = {}
    symbols = resolution.get("symbols", {})
    if not isinstance(symbols, dict):
        return by_source

    for symbol, entry in symbols.items():
        if not isinstance(entry, dict):
            continue
        defined_in = entry.get("defined_in", [])
        if not isinstance(defined_in, list):
            continue
        for anchor in defined_in:
            if not isinstance(anchor, str) or not anchor.strip():
                continue
            by_source.setdefault(anchor_source(anchor), set()).add(str(symbol))
    return by_source


def source_anchors_for_resolution(resolution: Dict[str, object]) -> Dict[str, list[str]]:
    anchors_by_source: Dict[str, list[str]] = {}
    symbols = resolution.get("symbols", {})
    if not isinstance(symbols, dict):
        return anchors_by_source

    for entry in symbols.values():
        if not isinstance(entry, dict):
            continue
        defined_in = entry.get("defined_in", [])
        if not isinstance(defined_in, list):
            continue
        for anchor in defined_in:
            if not isinstance(anchor, str) or not anchor.strip():
                continue
            source = anchor_source(anchor)
            anchors_by_source.setdefault(source, [])
            if anchor not in anchors_by_source[source]:
                anchors_by_source[source].append(anchor)
    return anchors_by_source


def fallback_definition_chunk(
    source: str,
    anchors: list[str],
    needed_symbols: set[str],
    table: Dict[str, Any],
    chunk_id: int,
) -> Dict[str, object] | None:
    path = Path(source)
    if not path.exists() or not path.is_file():
        return None

    raw = path.read_text(encoding="utf-8", errors="ignore")
    sections = [markdown_section(raw, anchor_fragment(anchor)) for anchor in anchors]
    text = "\n\n".join(section for section in dict.fromkeys(sections) if section.strip()).strip()
    if not text:
        text = raw.strip()

    text_symbols = {match.symbol for match in extract_symbols(text, table)}
    defined_here = needed_symbols.intersection(text_symbols) or set(needed_symbols)
    return {
        "source": source,
        "chunk_id": chunk_id,
        "text": text,
        "ted_symbols": sorted(defined_here),
        "ted_role": "definition",
        "ted_source": "file_fallback",
    }


def find_definition_chunks(
    chunks: List[Dict[str, object]],
    resolution: Dict[str, object],
    table: Dict[str, Any],
    limit: int,
) -> List[Tuple[float, Dict[str, object]]]:
    if limit <= 0:
        return []

    needed_by_source = symbols_defined_by_source(resolution)
    anchors_by_source = source_anchors_for_resolution(resolution)
    if not needed_by_source:
        return []

    depths = dependency_depths(resolution)
    candidates: List[Tuple[float, Dict[str, object], int]] = []
    found_sources: set[str] = set()

    for chunk in chunks:
        source = str(chunk.get("source", "")).replace("\\", "/")
        needed_symbols = needed_by_source.get(source)
        if not needed_symbols:
            continue

        text = str(chunk.get("text", ""))
        chunk_symbols = {match.symbol for match in extract_symbols(text, table)}
        defined_here = needed_symbols.intersection(chunk_symbols)
        if not defined_here and int(chunk.get("chunk_id", -1)) == 0:
            defined_here = set(needed_symbols)
        if not defined_here:
            continue

        found_sources.add(source)
        max_depth = max(depths.get(symbol, 0) for symbol in defined_here)
        score = 10.0 + max_depth + (0.25 if source.startswith("rag/seed_context/") else 0.0)
        expanded = dict(chunk)
        expanded["ted_symbols"] = sorted(defined_here)
        expanded["ted_role"] = "definition"
        expanded["ted_depth"] = max_depth
        expanded["ted_source"] = "index"
        candidates.append((score, expanded, max_depth))

    fallback_id = -1
    for source, needed_symbols in needed_by_source.items():
        if source in found_sources:
            continue
        chunk = fallback_definition_chunk(
            source,
            anchors_by_source.get(source, []),
            needed_symbols,
            table,
            fallback_id,
        )
        fallback_id -= 1
        if chunk is None:
            continue
        max_depth = max(depths.get(symbol, 0) for symbol in needed_symbols)
        chunk["ted_depth"] = max_depth
        score = 10.0 + max_depth + (0.25 if source.startswith("rag/seed_context/") else 0.0)
        candidates.append((score, chunk, max_depth))

    candidates.sort(
        key=lambda item: (
            -item[2],
            -item[0],
            str(item[1].get("source", "")),
            int(item[1].get("chunk_id", -1)),
        )
    )
    return [(score, chunk) for score, chunk, _ in candidates[:limit]]


def ted_expand_and_order(
    ranked: List[Tuple[float, Dict[str, object]]],
    chunks: List[Dict[str, object]],
    resolution: Dict[str, object],
    table: Dict[str, Any],
    expansion_limit: int,
) -> List[Tuple[float, Dict[str, object]]]:
    definition_chunks = find_definition_chunks(chunks, resolution, table, expansion_limit)
    depths = dependency_depths(resolution)
    defined_sources = symbols_defined_by_source(resolution)

    merged: Dict[tuple[str, int], Tuple[float, Dict[str, object]]] = {}
    for score, chunk in ranked:
        copied = dict(chunk)
        source = str(copied.get("source", "")).replace("\\", "/")
        symbols_for_source = defined_sources.get(source, set())
        if symbols_for_source:
            chunk_symbols = {match.symbol for match in extract_symbols(str(copied.get("text", "")), table)}
            defined_here = symbols_for_source.intersection(chunk_symbols)
            if defined_here:
                copied["ted_symbols"] = sorted(defined_here)
                copied["ted_role"] = "definition"
                copied["ted_depth"] = max(depths.get(symbol, 0) for symbol in defined_here)
        merged[chunk_key(copied)] = (score, copied)

    for score, chunk in definition_chunks:
        key = chunk_key(chunk)
        if key not in merged or score > merged[key][0]:
            merged[key] = (score, chunk)

    ordered = list(merged.values())
    ordered.sort(
        key=lambda item: (
            0 if item[1].get("ted_role") == "definition" else 1,
            -int(item[1].get("ted_depth", 0)),
            -float(item[0]),
            str(item[1].get("source", "")),
            int(item[1].get("chunk_id", -1)),
        )
    )
    return ordered


def validate_symbol_dependencies(
    question: str,
    ranked: List[Tuple[float, Dict[str, object]]],
    symbol_table_path: Path,
    max_depth: int,
) -> Dict[str, object]:
    table = load_symbol_table(symbol_table_path)
    question_matches = extract_symbols(question, table)
    roots = [match.symbol for match in question_matches]
    resolution = resolve_dependencies(roots, table, max_depth=max_depth)

    context_text = "\n\n".join(str(chunk["text"]) for _, chunk in ranked)
    context_symbols = {match.symbol for match in extract_symbols(context_text, table)}

    required = set(resolution["symbols"])
    missing_from_context = sorted(required.difference(context_symbols))

    return {
        "matches": [
            {
                "symbol": match.symbol,
                "matched": match.matched,
                "start": match.start,
                "end": match.end,
            }
            for match in question_matches
        ],
        "roots": roots,
        "required": sorted(required),
        "context_symbols": sorted(context_symbols),
        "missing_from_context": missing_from_context,
        "fallback_representations": fallback_representations(missing_from_context, table),
        "missing_from_table": resolution["missing"],
        "resolution": resolution,
    }


def format_symbol_validation(validation: Dict[str, object]) -> str:
    roots = validation["roots"]
    if not roots:
        return "No TED-RAG symbols were detected in the question."

    lines = ["TED-RAG symbol validation:"]
    lines.append("- Query symbols: " + ", ".join(str(symbol) for symbol in roots))

    missing_table = validation["missing_from_table"]
    if missing_table:
        lines.append("- Missing from symbol table: " + ", ".join(str(symbol) for symbol in missing_table))

    missing_context = validation["missing_from_context"]
    if missing_context:
        lines.append("- Missing from retrieved context: " + ", ".join(str(symbol) for symbol in missing_context))
        fallback_items = validation.get("fallback_representations", {})
        if isinstance(fallback_items, dict) and fallback_items:
            formatted = []
            for symbol in missing_context:
                fallback = fallback_items.get(str(symbol), {})
                if not isinstance(fallback, dict):
                    continue
                representation = fallback.get("representation", symbol)
                source = fallback.get("source", "canonical")
                formatted.append(f"{symbol} -> {representation} ({source})")
            if formatted:
                lines.append("- Representation fallback: " + "; ".join(formatted))
    else:
        lines.append("- All resolved symbol dependencies were found in retrieved context.")

    return "\n".join(lines)


def build_prompt(
    question: str,
    ranked: List[Tuple[float, Dict[str, object]]],
    symbol_validation: Dict[str, object] | None = None,
    constraint_summary: dict[str, object] | None = None,
    constraint_summary_limit: int = 12,
    max_chunk_chars: int | None = None,
    strict_citations: bool = False,
) -> str:
    context_blocks = []
    for i, (_, chunk) in enumerate(ranked, start=1):
        source = chunk["source"]
        chunk_id = chunk["chunk_id"]
        text = str(chunk["text"])
        if max_chunk_chars is not None and max_chunk_chars > 0 and len(text) > max_chunk_chars:
            text = text[:max_chunk_chars].rstrip() + "\n[chunk truncated for prompt budget]"
        ted_meta = ""
        if chunk.get("ted_role"):
            symbols = ", ".join(str(symbol) for symbol in chunk.get("ted_symbols", []))
            ted_meta = (
                f" ted_role={chunk.get('ted_role')}"
                f" ted_depth={chunk.get('ted_depth')}"
                f" ted_source={chunk.get('ted_source', 'index')}"
                f" ted_symbols={symbols}"
            )
        alpha_meta = ""
        if chunk.get("alpha_score"):
            alpha = chunk["alpha_score"]
            alpha_meta = (
                f" alpha_total={float(alpha['total']):.4f}"
                f" alpha_base={float(alpha['base']):.4f}"
                f" alpha_entanglement={float(alpha['entanglement']):.4f}"
                f" alpha_tachyon_risk={float(alpha['tachyon_risk']):.4f}"
                f" alpha_tachyon_novelty={float(alpha['tachyon_novelty']):.4f}"
            )
        stability_meta = ""
        if chunk.get("stability_score"):
            stability = chunk["stability_score"]
            stability_meta = (
                f" stability_adjusted={float(stability['adjusted']):.4f}"
                f" eta={float(stability['eta']):.4f}"
                f" gamma_susy={float(stability['gamma_susy']):.4f}"
                f" D_f={float(stability['fractal_dimension']):.4f}"
                f" N_SUSY={int(stability['unresolved_symbols'])}"
            )
        constraint_meta = ""
        if chunk.get("constraint"):
            constraint = chunk["constraint"]
            constraint_meta = (
                f" constraint_term={constraint.get('term')}"
                f" constraint_weight={float(constraint['weight']):.6f}"
                f" constraint_role={constraint.get('role')}"
            )
        context_blocks.append(
            f"[{i}] source={source} chunk={chunk_id}{ted_meta}{alpha_meta}{stability_meta}{constraint_meta}\n{text}"
        )

    context = "\n\n".join(context_blocks)
    symbol_note = ""
    if symbol_validation is not None:
        symbol_note = f"\n\nSymbol Validation:\n{format_symbol_validation(symbol_validation)}"
    constraint_note = ""
    if constraint_summary is not None:
        constraint_note = (
            "\n\n"
            + format_constraint_summary(constraint_summary, max_terms=constraint_summary_limit)
        )

    if strict_citations:
        return (
            "Context:\n"
            f"{context}\n\n"
            "RULES:\n"
            "1. Every sentence must end with [1], [2], or another listed context citation.\n"
            "2. If you cannot cite, output ⠏ instead of guessing.\n"
            "3. Use Ψ' not Φ'.\n"
            "4. Only use symbols defined in Context.\n\n"
            f"Question: {question}{symbol_note}{constraint_note}\n"
            "Answer format:\n"
            "- Bullet 1 [1]\n"
            "- Bullet 2 [1]\n"
            "- Bullet 3 [2]\n\n"
            "Answer:"
        )

    return (
        "You are a rigorous math-and-code research assistant for the TED-RAG system. "
        "Answer only from the retrieved context and clearly mark uncertainty.\n\n"
        "Rules:\n"
        "1) Show derivations step by step when doing math.\n"
        "2) If evidence is missing, say exactly what is missing.\n"
        "3) Cite claims with [n] references from context items.\n"
        "4) Do not invent equations, files, or symbols not in context.\n\n"
        "When context items include `ted_role=definition`, use them as definition anchors before interpreting dependent context.\n\n"
        "When context items include `alpha_*` fields, treat them as experimental retrieval diagnostics, not source evidence.\n\n"
        "When context items include `stability_*`, `eta`, `gamma_susy`, `D_f`, or `N_SUSY`, treat them as experimental retrieval diagnostics, not source evidence.\n\n"
        "When a Fractal Constraint is supplied, treat it as the retrieval weighting plan for the context window; cite the underlying context items, not the weight metadata itself.\n\n"
        "If symbol validation lists a representation fallback, use it only as an explicit fallback for a missing or unrenderable canonical symbol.\n\n"
        "If a `rag/seed_context` source appears, treat it as canonical orientation context while still citing broader corpus evidence separately.\n\n"
        f"Question:\n{question}{symbol_note}{constraint_note}\n\n"
        f"Context:\n{context}\n\n"
        "Answer with sections: Summary, Reasoning, Citations."
    )


def ollama_options(
    num_predict: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
) -> Dict[str, object]:
    options: Dict[str, object] = {}
    if num_predict is not None:
        options["num_predict"] = num_predict
    if temperature is not None:
        options["temperature"] = temperature
    if stop:
        options["stop"] = stop
    return options


def generate_answer(
    base_url: str,
    model: str,
    prompt: str,
    num_predict: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
    timeout: int = 240,
) -> str:
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    payload: Dict[str, object] = {"model": model, "prompt": prompt, "stream": False}
    options = ollama_options(num_predict=num_predict, temperature=temperature, stop=stop)
    if options:
        payload["options"] = options
    response = requests.post(
        endpoint,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return str(data.get("response", "")).strip()


def stream_answer(
    base_url: str,
    model: str,
    prompt: str,
    num_predict: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
) -> str:
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    payload: Dict[str, object] = {"model": model, "prompt": prompt, "stream": True}
    options = ollama_options(num_predict=num_predict, temperature=temperature, stop=stop)
    if options:
        payload["options"] = options

    chunks: list[str] = []
    with requests.post(endpoint, json=payload, stream=True, timeout=(10, None)) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            piece = str(data.get("response", ""))
            if piece:
                print(piece, end="", flush=True)
                chunks.append(piece)
            if data.get("done"):
                break
    print()
    return "".join(chunks).strip()


def line_has_citation(line: str) -> bool:
    return bool(re.search(r"\[\d+\]\s*$", line.strip()))


def inject_missing_citations(answer: str, ranked: List[Tuple[float, Dict[str, object]]]) -> tuple[str, bool]:
    fallback_ref = "[1]" if ranked else "⠏"
    changed = False
    fixed_lines: list[str] = []

    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            fixed_lines.append(line)
            continue
        if stripped.startswith("===") or stripped.startswith("#"):
            fixed_lines.append(line)
            continue
        if re.fullmatch(r"\[\d+\].*", stripped):
            fixed_lines.append(line)
            continue
        if line_has_citation(stripped):
            fixed_lines.append(line)
            continue
        fixed_lines.append(line.rstrip() + f" {fallback_ref}")
        changed = True

    fixed = "\n".join(fixed_lines)
    if not re.search(r"\[\d+\]", fixed) and ranked:
        fixed += f"\n\n[1] {ranked[0][1]['source']} (weight={ranked[0][1].get('constraint', {}).get('weight', 1.0):.6f})"
        changed = True
    return fixed, changed


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    if args.fast:
        args = streaming_potato_controls(args)
    timings: Dict[str, float] = {}
    profile_to_index = {
        "default": args.index,
        "quick": "rag/index_quick",
        "medium": "rag/index_medium",
        "deep": "rag/index_deep",
    }
    index_dir = Path(profile_to_index[args.profile])

    t0 = time.perf_counter()
    embeddings, chunks = load_index(index_dir)
    timings["load_index"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    q_vec = embed_query(args.ollama_url, args.embed_model, args.question)
    timings["embed_query"] = time.perf_counter() - t0
    table = load_symbol_table(Path(args.symbol_table))
    t0 = time.perf_counter()
    ranked = retrieve(
        args.question,
        q_vec,
        embeddings,
        chunks,
        args.top_k,
        symbol_table=table,
        alpha_enabled=args.alpha_score,
        alpha_candidates=args.alpha_candidates,
        alpha_entanglement_sample=args.alpha_entanglement_sample,
        alpha_lambda_risk=args.alpha_lambda_risk,
        alpha_lambda_novelty=args.alpha_lambda_novelty,
    )
    timings["retrieve"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    symbol_validation = validate_symbol_dependencies(
        args.question,
        ranked,
        Path(args.symbol_table),
        args.symbol_depth,
    )
    timings["symbol_validation_initial"] = time.perf_counter() - t0

    if not args.no_ted_expand and symbol_validation["roots"]:
        t0 = time.perf_counter()
        ranked = ted_expand_and_order(
            ranked,
            chunks,
            symbol_validation["resolution"],
            table,
            args.ted_expansion_limit,
        )
        symbol_validation = validate_symbol_dependencies(
            args.question,
            ranked,
            Path(args.symbol_table),
            args.symbol_depth,
        )
        timings["ted_expand"] = time.perf_counter() - t0

    if args.stability_rerank:
        t0 = time.perf_counter()
        ranked = stability_rerank(
            ranked,
            table,
            eta0=args.eta0,
            eta_alpha=args.eta_alpha,
            gamma0=args.gamma0,
            gamma_beta=args.gamma_beta,
            preserve_ted_definitions=True,
        )
        symbol_validation = validate_symbol_dependencies(
            args.question,
            ranked,
            Path(args.symbol_table),
            args.symbol_depth,
        )
        timings["stability_rerank"] = time.perf_counter() - t0

    constraint_summary = None
    if args.fractal_context:
        t0 = time.perf_counter()
        ranked, constraint_summary = build_fractal_constraint(
            ranked,
            weight_floor=args.constraint_weight_floor,
        )
        timings["fractal_context"] = time.perf_counter() - t0

    if symbol_validation["roots"]:
        print("=== Symbol Validation ===")
        print(format_symbol_validation(symbol_validation))
        print()

    if args.require_symbols and (
        symbol_validation["missing_from_context"] or symbol_validation["missing_from_table"]
    ):
        raise SystemExit("Required TED-RAG symbol dependencies are missing from retrieved context.")

    if args.show_context:
        print("=== Retrieved Context ===")
        for i, (score, chunk) in enumerate(ranked, start=1):
            print(f"[{i}] score={score:.4f} source={chunk['source']} chunk={chunk['chunk_id']}")
            if chunk.get("ted_role"):
                print(
                    "TED "
                    f"role={chunk.get('ted_role')} "
                    f"depth={chunk.get('ted_depth')} "
                    f"source={chunk.get('ted_source', 'index')} "
                    f"symbols={', '.join(str(symbol) for symbol in chunk.get('ted_symbols', []))}"
                )
            if chunk.get("alpha_score"):
                alpha = chunk["alpha_score"]
                print(
                    "Alpha "
                    f"total={float(alpha['total']):.4f} "
                    f"base={float(alpha['base']):.4f} "
                    f"ent={float(alpha['entanglement']):.4f} "
                    f"tachyon={float(alpha['tachyon_risk']):.4f} "
                    f"novelty={float(alpha['tachyon_novelty']):.4f} "
                    f"lambda_risk={float(alpha['lambda_risk']):.4f} "
                    f"lambda_novelty={float(alpha['lambda_novelty']):.4f}"
                )
            if chunk.get("stability_score"):
                stability = chunk["stability_score"]
                print(
                    "Stability "
                    f"adjusted={float(stability['adjusted']):.4f} "
                    f"eta={float(stability['eta']):.4f} "
                    f"gamma={float(stability['gamma_susy']):.4f} "
                    f"D_f={float(stability['fractal_dimension']):.4f} "
                    f"N_SUSY={int(stability['unresolved_symbols'])}"
                )
            if chunk.get("constraint"):
                constraint = chunk["constraint"]
                print(
                    "Constraint "
                    f"term={constraint.get('term')} "
                    f"weight={float(constraint['weight']):.6f} "
                    f"role={constraint.get('role')}"
                )
            print(chunk["text"][:350].replace("\n", " "))
            print()

    prompt = build_prompt(
        args.question,
        ranked,
        symbol_validation,
        constraint_summary=constraint_summary,
        constraint_summary_limit=args.constraint_summary_limit,
        max_chunk_chars=args.max_chunk_chars,
        strict_citations=args.strict_citations,
    )
    timings["prompt_chars"] = float(len(prompt))
    if constraint_summary is not None:
        print("=== Fractal Constraint ===")
        print(format_constraint_summary(constraint_summary, max_terms=args.constraint_summary_limit))
        print()
    print("=== Answer ===")
    t0 = time.perf_counter()
    if args.stream:
        answer = stream_answer(
            args.ollama_url,
            args.llm_model,
            prompt,
            num_predict=args.num_predict,
            temperature=args.temperature,
            stop=args.stop,
        )
    else:
        answer = generate_answer(
            args.ollama_url,
            args.llm_model,
            prompt,
            num_predict=args.num_predict,
            temperature=args.temperature,
            stop=args.stop,
            timeout=args.generation_timeout,
        )
    if args.strict_citations:
        fixed_answer, citation_changed = inject_missing_citations(answer, ranked)
        if citation_changed:
            if args.stream:
                print("\n=== Citation Pass ===")
            print(fixed_answer)
            answer = fixed_answer
        elif not args.stream:
            print(answer)
    elif not args.stream:
        print(answer)
    timings["ollama_generate"] = time.perf_counter() - t0
    timings["answer_chars"] = float(len(answer))
    if args.print_timing:
        print("\n=== Timing ===")
        for key, value in timings.items():
            if key.endswith("_chars"):
                print(f"{key}={int(value)}")
            else:
                print(f"{key}={value:.3f}s")
    print("\n=== Citation Map ===")
    for i, (_, chunk) in enumerate(ranked, start=1):
        print(f"[{i}] {chunk['source']} (chunk {chunk['chunk_id']})")


if __name__ == "__main__":
    main()
