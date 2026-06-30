#!/usr/bin/env python3
"""Index Chronomere conductor for compiler-validated TED-RAG profiles."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_LOG = ROOT / "Index" / "session_logs" / "chronomere_sessions.jsonl"
DEFAULT_MEMORY_LOG = ROOT / "index_chronomere" / "memory" / "root.jsonl"
DEFAULT_MEMORY_DB = ROOT / "index_chronomere" / "memory" / "chronomere_memory.sqlite"
UTF8_ENV = {
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}
REQUIRED_INDEX_FILES = ("embeddings.npy", "chunks.jsonl")
INDEX_TIMESTAMP_FILES = ("embeddings.npy", "chunks.jsonl", "manifest.json")
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".py", ".ps1"}
DEFAULT_EXCLUDE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".pdf", ".bin", ".pkl"}
DEFAULT_INCLUDE_PATHS = ["rag/seed_context", "docs", "genesis_public"]
PROFILE_MAX_CHUNKS = {
    "quick": 100,
    "medium": 500,
    "deep": 4000,
}
EMBEDDING_ONLY_MODEL = "nomic-embed-text"
PROFILE_INDEXES: Dict[str, Path] = {
    "quick": ROOT / "rag" / "index_quick",
    "medium": ROOT / "rag" / "index_medium",
    "deep": ROOT / "rag" / "index_deep",
}
CONTEXT_BUDGETS: Dict[str, Dict[str, int]] = {
    "potato": {
        "top_k": 2,
        "max_chunk_chars": 600,
        "num_predict": 120,
        "constraint_summary_limit": 2,
    },
    "quick": {
        "top_k": 4,
        "max_chunk_chars": 700,
        "num_predict": 256,
        "constraint_summary_limit": 4,
    },
    "medium": {
        "top_k": 8,
        "max_chunk_chars": 900,
        "num_predict": 512,
        "constraint_summary_limit": 8,
    },
    "deep": {
        "top_k": 12,
        "max_chunk_chars": 1200,
        "num_predict": 900,
        "constraint_summary_limit": 12,
    },
    "research": {
        "top_k": 12,
        "max_chunk_chars": 1200,
        "num_predict": 900,
        "constraint_summary_limit": 12,
    },
}
TOOL_REGISTRY: Dict[str, str] = {
    "query": "Route a question through TED-RAG using Chronomere mode, budget, audit, and logging.",
    "check-indexes": "Check quick, medium, and deep index files.",
    "rebuild-advice": "Report which indexes are missing or stale and print rebuild commands.",
    "resolve-symbols": "Resolve TED-RAG symbols and dependencies from question text.",
    "show-context": "Run a query with retrieved context displayed.",
    "audit-citations": "Run a query with citation auditing enabled.",
    "remember": "Save a curated Chronomere memory atom.",
    "recall-memory": "Recall curated Chronomere memory atoms by lexical overlap.",
}


def configure_utf8_runtime() -> None:
    os.environ.update(UTF8_ENV)
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.update(UTF8_ENV)
    return env


QUICK_TERMS = {
    "define",
    "summary",
    "summarize",
    "quick",
    "brief",
    "short",
    "bullet",
    "bullets",
}

DEEP_TERMS = {
    "audit",
    "contradiction",
    "contradictions",
    "derive",
    "deep",
    "exhaustive",
    "full",
    "map",
    "research",
    "synthesize",
    "unify",
}

MEDIUM_TERMS = {
    "compare",
    "connect",
    "explain",
    "relationship",
    "trace",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route a question through the right compiler-validated TED-RAG profile."
    )
    parser.add_argument("--question", "--query", dest="question", help="Question to ask TED-RAG")
    parser.add_argument(
        "--tool",
        choices=sorted(TOOL_REGISTRY),
        default="query",
        help="Named Chronomere tool to run",
    )
    parser.add_argument("--list-tools", action="store_true", help="List Chronomere tools and exit")
    parser.add_argument(
        "--mode",
        choices=["auto", "potato", "quick", "medium", "deep", "research"],
        default="auto",
        help="Conductor mode. auto picks from the question.",
    )
    parser.add_argument("--show-context", action="store_true", help="Show retrieved context")
    parser.add_argument("--stream", action="store_true", help="Stream generation tokens")
    parser.add_argument("--print-timing", action="store_true", help="Print timing diagnostics")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument(
        "--skip-ollama-check",
        action="store_true",
        help="Skip Chronomere's Ollama reachability preflight",
    )
    parser.add_argument(
        "--escalate",
        action="store_true",
        help="Start low-cost and retry with deeper modes when the answer looks weak",
    )
    parser.add_argument(
        "--max-escalation",
        choices=["quick", "medium", "research"],
        default="research",
        help="Deepest mode allowed by --escalate",
    )
    parser.add_argument(
        "--audit-citations",
        action="store_true",
        help="Audit final answer citations after a non-streaming query",
    )
    parser.add_argument(
        "--no-auto-audit",
        action="store_true",
        help="Disable automatic citation audit for strict-citation modes",
    )
    parser.add_argument("--top-k", type=int, help="Override retrieved chunk count")
    parser.add_argument("--max-chunk-chars", type=int, help="Override maximum characters per chunk")
    parser.add_argument("--num-predict", type=int, help="Override generation token budget")
    parser.add_argument(
        "--constraint-summary-limit",
        type=int,
        help="Override fractal constraint summary term count",
    )
    parser.add_argument(
        "--check-indexes",
        action="store_true",
        help="Deprecated alias for --tool check-indexes",
    )
    parser.add_argument("--symbol-depth", type=int, default=1, help="Dependency depth for resolve-symbols")
    parser.add_argument(
        "--skip-index-check",
        action="store_true",
        help="Skip Chronomere's preflight index sentinel",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rag/query.py command without running it",
    )
    parser.add_argument(
        "--session-log",
        default=str(DEFAULT_SESSION_LOG),
        help="JSONL path for Chronomere session records",
    )
    parser.add_argument(
        "--no-session-log",
        action="store_true",
        help="Disable Chronomere session logging",
    )
    parser.add_argument(
        "--memory-path",
        default=str(DEFAULT_MEMORY_LOG),
        help="JSONL seed/export path for curated Chronomere memories",
    )
    parser.add_argument(
        "--memory-db",
        default=str(DEFAULT_MEMORY_DB),
        help="SQLite path for durable Chronomere memory",
    )
    parser.add_argument("--memory-title", help="Title for --tool remember")
    parser.add_argument("--memory-note", help="Memory text for --tool remember or recall query text")
    parser.add_argument("--memory-kind", default="decision", help="Memory kind for --tool remember")
    parser.add_argument("--memory-tags", help="Comma-separated tags for --tool remember")
    parser.add_argument("--memory-limit", type=int, default=5, help="Maximum memories to recall")
    parser.add_argument("--memory-embed-model", default=EMBEDDING_ONLY_MODEL, help="Ollama model for memory vectors")
    parser.add_argument(
        "--memory-vector-recall",
        action="store_true",
        help="Opt into numpy vector recall over curated memory atoms",
    )
    parser.add_argument(
        "--memory-vectorize",
        action="store_true",
        help="Store/update a vector for --tool remember",
    )
    parser.add_argument(
        "--use-memory",
        action="store_true",
        help="Fold recalled Chronomere memory into a query prompt",
    )
    return parser.parse_args()


def print_tool_registry() -> None:
    print("Chronomere tools:")
    for name in sorted(TOOL_REGISTRY):
        print(f"- {name}: {TOOL_REGISTRY[name]}")


def selected_tool(args: argparse.Namespace) -> str:
    if args.check_indexes:
        return "check-indexes"
    return args.tool


def validate_numeric_args(args: argparse.Namespace) -> List[str]:
    checks = {
        "top_k": (getattr(args, "top_k", None), 1),
        "max_chunk_chars": (getattr(args, "max_chunk_chars", None), 1),
        "num_predict": (getattr(args, "num_predict", None), 1),
        "constraint_summary_limit": (getattr(args, "constraint_summary_limit", None), 1),
        "symbol_depth": (getattr(args, "symbol_depth", None), 0),
        "memory_limit": (getattr(args, "memory_limit", None), 1),
    }
    problems: List[str] = []
    for name, (value, minimum) in checks.items():
        if value is not None and value < minimum:
            problems.append(f"--{name.replace('_', '-')} must be >= {minimum}")
    return problems


def ollama_is_reachable(base_url: str, timeout: float = 2.0) -> Tuple[bool, str]:
    endpoint = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        return False, str(exc)
    return True, endpoint


def print_preflight_problems(problems: Iterable[str]) -> None:
    print("[Chronomere] preflight failed:")
    for problem in problems:
        print(f"  - {problem}")


def preflight_before_tool_call(args: argparse.Namespace, tool: str, needs_ollama: bool) -> bool:
    problems = validate_numeric_args(args)
    ollama_problem = False
    if needs_ollama and not args.dry_run and not args.skip_ollama_check:
        ok, detail = ollama_is_reachable(args.ollama_url)
        if not ok:
            ollama_problem = True
            problems.append(f"Ollama is not reachable at {args.ollama_url}: {detail}")

    if problems:
        print_preflight_problems(problems)
        if ollama_problem:
            print("[Chronomere] use --skip-ollama-check to bypass only the Ollama reachability check.")
        return False

    if needs_ollama and not args.dry_run and not args.skip_ollama_check:
        print(f"[Chronomere] Ollama preflight OK: {args.ollama_url}")
    return True


def parse_memory_tags(raw_tags: str | None) -> List[str]:
    if not raw_tags:
        return []
    tags = []
    for tag in raw_tags.split(","):
        normalized = tag.strip().lower().replace(" ", "-")
        if normalized:
            tags.append(normalized)
    return sorted(set(tags))


def memory_terms(text: str) -> Set[str]:
    return {
        term
        for term in re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if len(term) > 1
    }


def memory_text(record: Dict[str, object]) -> str:
    tags = record.get("tags", [])
    tag_text = " ".join(str(tag) for tag in tags) if isinstance(tags, list) else str(tags)
    return " ".join(
        str(record.get(key, ""))
        for key in ("title", "kind", "summary", "note")
    ) + " " + tag_text


def compact_text(text: str, limit: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def build_memory_record(args: argparse.Namespace) -> Dict[str, object]:
    note = str(args.memory_note or "").strip()
    title = str(args.memory_title or "").strip()
    if not title:
        title = compact_text(note, 80) or "Untitled memory"
    tags = parse_memory_tags(args.memory_tags)
    timestamp = datetime.now(timezone.utc).isoformat()
    memory_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48]
    if not memory_id:
        memory_id = "memory"

    return {
        "id": f"{memory_id}-{int(time.time())}",
        "timestamp_utc": timestamp,
        "kind": args.memory_kind,
        "title": title,
        "summary": compact_text(note),
        "note": note,
        "tags": tags,
    }


def write_memory_record(record: Dict[str, object], memory_path: Path) -> None:
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    with memory_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def read_memory_records(memory_path: Path) -> List[Dict[str, object]]:
    if not memory_path.is_file():
        return []

    records: List[Dict[str, object]] = []
    with memory_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def memory_db_path(args: argparse.Namespace | None = None) -> Path:
    if args is None:
        return DEFAULT_MEMORY_DB
    return Path(getattr(args, "memory_db", str(DEFAULT_MEMORY_DB)))


def init_memory_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    note TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'sqlite'
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp_utc)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    memory_id TEXT PRIMARY KEY,
                    embed_model TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    updated_utc TEXT NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
                """
            )


def normalize_memory_record(record: Dict[str, object]) -> Dict[str, object]:
    title = str(record.get("title") or "Untitled memory")
    note = str(record.get("note") or record.get("summary") or "")
    tags = record.get("tags", [])
    if not isinstance(tags, list):
        tags = parse_memory_tags(str(tags))
    normalized = {
        "id": str(record.get("id") or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "memory"),
        "timestamp_utc": str(record.get("timestamp_utc") or datetime.now(timezone.utc).isoformat()),
        "kind": str(record.get("kind") or "memory"),
        "title": title,
        "summary": str(record.get("summary") or compact_text(note)),
        "note": note,
        "tags": [str(tag) for tag in tags],
        "source": str(record.get("source") or "sqlite"),
    }
    return normalized


def write_memory_record_sqlite(record: Dict[str, object], db_path: Path, source: str = "sqlite") -> None:
    init_memory_db(db_path)
    normalized = normalize_memory_record({**record, "source": source})
    with closing(sqlite3.connect(db_path)) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, timestamp_utc, kind, title, summary, note, tags_json, search_text, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    timestamp_utc=excluded.timestamp_utc,
                    kind=excluded.kind,
                    title=excluded.title,
                    summary=excluded.summary,
                    note=excluded.note,
                    tags_json=excluded.tags_json,
                    search_text=excluded.search_text,
                    source=excluded.source
                """,
                (
                    normalized["id"],
                    normalized["timestamp_utc"],
                    normalized["kind"],
                    normalized["title"],
                    normalized["summary"],
                    normalized["note"],
                    json.dumps(normalized["tags"], ensure_ascii=True),
                    memory_text(normalized),
                    normalized["source"],
                ),
            )


def row_to_memory_record(row: sqlite3.Row) -> Dict[str, object]:
    try:
        tags = json.loads(row["tags_json"])
    except json.JSONDecodeError:
        tags = []
    if not isinstance(tags, list):
        tags = []
    return {
        "id": row["id"],
        "timestamp_utc": row["timestamp_utc"],
        "kind": row["kind"],
        "title": row["title"],
        "summary": row["summary"],
        "note": row["note"],
        "tags": tags,
        "source": row["source"],
    }


def read_memory_records_sqlite(db_path: Path) -> List[Dict[str, object]]:
    if not db_path.is_file():
        return []
    init_memory_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, timestamp_utc, kind, title, summary, note, tags_json, source
            FROM memories
            ORDER BY timestamp_utc ASC
            """
        ).fetchall()
    return [row_to_memory_record(row) for row in rows]


def embed_memory_text(base_url: str, model: str, text: str) -> np.ndarray:
    endpoint = f"{base_url.rstrip('/')}/api/embeddings"
    response = requests.post(
        endpoint,
        json={"model": model, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    vector = np.asarray(payload.get("embedding", []), dtype=np.float32)
    if vector.size == 0:
        raise RuntimeError("No embedding returned for memory text")
    norm = float(np.linalg.norm(vector))
    if norm:
        vector = vector / norm
    return vector


def write_memory_vector(db_path: Path, memory_id: str, model: str, vector: np.ndarray) -> None:
    init_memory_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO memory_vectors (memory_id, embed_model, vector_json, updated_utc)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    embed_model=excluded.embed_model,
                    vector_json=excluded.vector_json,
                    updated_utc=excluded.updated_utc
                """,
                (
                    memory_id,
                    model,
                    json.dumps(vector.astype(float).tolist(), ensure_ascii=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )


def read_memory_vectors(db_path: Path, model: str) -> Dict[str, np.ndarray]:
    if not db_path.is_file():
        return {}
    init_memory_db(db_path)
    vectors: Dict[str, np.ndarray] = {}
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT memory_id, vector_json
            FROM memory_vectors
            WHERE embed_model = ?
            """,
            (model,),
        ).fetchall()
    for memory_id, vector_json in rows:
        try:
            vector = np.asarray(json.loads(vector_json), dtype=np.float32)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if vector.size:
            vectors[str(memory_id)] = vector
    return vectors


def ensure_memory_vectors(args: argparse.Namespace, records: List[Dict[str, object]], db_path: Path) -> int:
    vectors = read_memory_vectors(db_path, args.memory_embed_model)
    created = 0
    for record in records:
        memory_id = str(record.get("id", ""))
        if not memory_id or memory_id in vectors:
            continue
        vector = embed_memory_text(args.ollama_url, args.memory_embed_model, memory_text(record))
        write_memory_vector(db_path, memory_id, args.memory_embed_model, vector)
        created += 1
    return created


def ensure_memory_db_seeded(memory_path: Path, db_path: Path) -> int:
    init_memory_db(db_path)
    records = read_memory_records(memory_path)
    imported = 0
    for record in records:
        normalized = normalize_memory_record(record)
        with closing(sqlite3.connect(db_path)) as conn:
            exists = conn.execute(
                "SELECT 1 FROM memories WHERE id = ?",
                (normalized["id"],),
            ).fetchone()
        if exists:
            continue
        write_memory_record_sqlite(normalized, db_path, source="jsonl-seed")
        imported += 1
    return imported


def score_memory_record(query: str, record: Dict[str, object]) -> int:
    query_terms = memory_terms(query)
    if not query_terms:
        return 0

    record_terms = memory_terms(memory_text(record))
    overlap = query_terms & record_terms
    score = len(overlap)
    title_terms = memory_terms(str(record.get("title", "")))
    score += len(query_terms & title_terms)
    tags = record.get("tags", [])
    tag_text = " ".join(str(tag) for tag in tags) if isinstance(tags, list) else str(tags)
    tag_terms = memory_terms(tag_text)
    score += len(query_terms & tag_terms)
    return score


def recall_memory_records(query: str, memory_path: Path, limit: int) -> List[Dict[str, object]]:
    scored: List[Tuple[int, str, Dict[str, object]]] = []
    for record in read_memory_records(memory_path):
        score = score_memory_record(query, record)
        if score <= 0:
            continue
        scored.append((score, str(record.get("timestamp_utc", "")), record))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [record for _, _, record in scored[:limit]]


def recall_memory_records_sqlite(query: str, memory_path: Path, db_path: Path, limit: int) -> List[Dict[str, object]]:
    ensure_memory_db_seeded(memory_path, db_path)
    query_terms = sorted(memory_terms(query))
    sql_candidates: List[Dict[str, object]] = []
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        if query_terms:
            where = " OR ".join("search_text LIKE ?" for _ in query_terms)
            params = [f"%{term}%" for term in query_terms]
            rows = conn.execute(
                f"""
                SELECT id, timestamp_utc, kind, title, summary, note, tags_json, source
                FROM memories
                WHERE {where}
                ORDER BY timestamp_utc DESC
                """,
                params,
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, timestamp_utc, kind, title, summary, note, tags_json, source
                FROM memories
                ORDER BY timestamp_utc DESC
                """
            ).fetchall()
    sql_candidates = [row_to_memory_record(row) for row in rows]

    scored: List[Tuple[int, str, Dict[str, object]]] = []
    for record in sql_candidates:
        score = score_memory_record(query, record)
        if score <= 0:
            continue
        scored.append((score, str(record.get("timestamp_utc", "")), record))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [record for _, _, record in scored[:limit]]


def recall_memory_records_hybrid(args: argparse.Namespace, query: str) -> List[Dict[str, object]]:
    memory_path = Path(args.memory_path)
    db_path = memory_db_path(args)
    ensure_memory_db_seeded(memory_path, db_path)
    if not getattr(args, "memory_vector_recall", False) or getattr(args, "dry_run", False):
        return recall_memory_records_sqlite(query, memory_path, db_path, args.memory_limit)

    records = read_memory_records_sqlite(db_path)
    ensure_memory_vectors(args, records, db_path)
    vectors = read_memory_vectors(db_path, args.memory_embed_model)
    query_vector = embed_memory_text(args.ollama_url, args.memory_embed_model, query)
    scored: List[Tuple[float, str, Dict[str, object]]] = []
    for record in records:
        memory_id = str(record.get("id", ""))
        vector = vectors.get(memory_id)
        if vector is None or vector.size != query_vector.size:
            continue
        lexical_score = float(score_memory_record(query, record))
        vector_score = float(np.dot(query_vector, vector))
        combined = lexical_score + vector_score
        if combined <= 0:
            continue
        scored.append((combined, str(record.get("timestamp_utc", "")), record))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [record for _, _, record in scored[: args.memory_limit]]


def memory_context_for_query(question: str, memories: List[Dict[str, object]]) -> str:
    if not memories:
        return question

    lines = ["Chronomere memory context:"]
    for record in memories:
        title = compact_text(str(record.get("title", "Untitled memory")), 80)
        summary = compact_text(str(record.get("summary") or record.get("note") or ""), 220)
        tags = record.get("tags", [])
        tag_text = ", ".join(str(tag) for tag in tags) if isinstance(tags, list) else str(tags)
        suffix = f" tags={tag_text}" if tag_text else ""
        lines.append(f"- {title}: {summary}{suffix}")
    lines.extend(["", "Current question:", question])
    return "\n".join(lines)


def recalled_memory_for_args(args: argparse.Namespace) -> List[Dict[str, object]]:
    if not getattr(args, "use_memory", False):
        return []
    return recall_memory_records_hybrid(args, args.question)


def print_memory_records(records: List[Dict[str, object]]) -> None:
    if not records:
        print("[Chronomere] no matching memories found.")
        return

    print(f"[Chronomere] recalled {len(records)} memory atom(s):")
    for record in records:
        title = compact_text(str(record.get("title", "Untitled memory")), 90)
        kind = str(record.get("kind", "memory"))
        memory_id = str(record.get("id", "unknown"))
        summary = compact_text(str(record.get("summary") or record.get("note") or ""), 220)
        print(f"- {title} ({kind}, {memory_id})")
        if summary:
            print(f"  {summary}")


def choose_mode(question: str) -> str:
    normalized = question.lower()
    tokens = {part.strip(".,:;!?()[]{}\"'") for part in normalized.split()}

    if tokens & DEEP_TERMS:
        return "research"
    if tokens & MEDIUM_TERMS:
        return "medium"
    if tokens & QUICK_TERMS:
        return "quick"
    if len(tokens) <= 8:
        return "quick"
    return "medium"


def profiles_for_mode(mode: str) -> List[str]:
    if mode in {"potato", "quick"}:
        return ["quick"]
    if mode == "medium":
        return ["medium"]
    if mode in {"deep", "research"}:
        return ["deep"]
    raise ValueError(f"Unknown Chronomere mode: {mode}")


def escalation_plan(start_mode: str, max_mode: str) -> List[str]:
    order = ["quick", "medium", "research"]
    normalized_start = "quick" if start_mode == "potato" else start_mode
    if normalized_start == "deep":
        normalized_start = "research"
    if normalized_start not in order:
        normalized_start = "medium"
    if max_mode not in order:
        max_mode = "research"

    start_idx = order.index(normalized_start)
    max_idx = order.index(max_mode)
    if start_idx > max_idx:
        return [normalized_start]
    return order[start_idx : max_idx + 1]


def check_indexes(profiles: Iterable[str]) -> List[str]:
    problems: List[str] = []
    for profile in profiles:
        index_dir = PROFILE_INDEXES[profile]
        if not index_dir.is_dir():
            problems.append(f"{profile}: missing directory {index_dir}")
            continue

        for filename in REQUIRED_INDEX_FILES:
            path = index_dir / filename
            if not path.is_file():
                problems.append(f"{profile}: missing {path}")

    return problems


def print_index_report(profiles: Iterable[str]) -> bool:
    profiles = list(profiles)
    problems = check_indexes(profiles)

    if problems:
        print("[Chronomere] index sentinel found missing files:")
        for problem in problems:
            print(f"  - {problem}")
        print("[Chronomere] rebuild suggestion: run rag/build_index.py for the missing profile.")
        return False

    names = ", ".join(profiles)
    print(f"[Chronomere] index sentinel OK: {names}")
    return True


def iter_corpus_files(
    root: Path = ROOT,
    include_paths: Iterable[str] = DEFAULT_INCLUDE_PATHS,
    exclude_extensions: Set[str] = DEFAULT_EXCLUDE_EXTENSIONS,
) -> Iterable[Path]:
    for rel in include_paths:
        target = (root / rel).resolve()
        if not target.exists():
            continue
        if target.is_file():
            suffix = target.suffix.lower()
            if suffix in TEXT_EXTENSIONS and suffix not in exclude_extensions:
                yield target
            continue
        for path in target.rglob("*"):
            suffix = path.suffix.lower()
            if path.is_file() and suffix in TEXT_EXTENSIONS and suffix not in exclude_extensions:
                yield path


def latest_source_mtime(root: Path = ROOT) -> Tuple[float | None, Path | None, int]:
    newest_mtime: float | None = None
    newest_path: Path | None = None
    count = 0
    for path in iter_corpus_files(root=root):
        count += 1
        mtime = path.stat().st_mtime
        if newest_mtime is None or mtime > newest_mtime:
            newest_mtime = mtime
            newest_path = path
    return newest_mtime, newest_path, count


def index_timestamp(index_dir: Path) -> Tuple[float | None, List[Path]]:
    missing = [index_dir / filename for filename in REQUIRED_INDEX_FILES if not (index_dir / filename).is_file()]
    if missing:
        return None, missing

    present = [index_dir / filename for filename in INDEX_TIMESTAMP_FILES if (index_dir / filename).is_file()]
    if not present:
        return None, [index_dir / REQUIRED_INDEX_FILES[0]]
    return min(path.stat().st_mtime for path in present), []


def rebuild_command_for_profile(profile: str) -> List[str]:
    return [
        sys.executable,
        "-u",
        str(ROOT / "rag" / "build_index.py"),
        "--root",
        ".",
        "--out",
        f"rag/index_{profile}",
        "--max-chunks",
        str(PROFILE_MAX_CHUNKS[profile]),
        "--embed-model",
        EMBEDDING_ONLY_MODEL,
    ]


def rebuild_advice() -> Tuple[bool, List[Dict[str, object]]]:
    source_mtime, source_path, source_count = latest_source_mtime()
    reports: List[Dict[str, object]] = []

    for profile, index_dir in PROFILE_INDEXES.items():
        idx_mtime, missing = index_timestamp(index_dir)
        if missing:
            status = "missing"
            reason = "missing required index files"
        elif source_mtime is None:
            status = "unknown"
            reason = "no source corpus files found"
        elif idx_mtime is not None and source_mtime > idx_mtime:
            status = "stale"
            reason = "source corpus is newer than index"
        else:
            status = "fresh"
            reason = "index is newer than corpus sources"

        reports.append(
            {
                "profile": profile,
                "status": status,
                "reason": reason,
                "index_dir": index_dir,
                "index_mtime": idx_mtime,
                "missing": missing,
                "source_mtime": source_mtime,
                "source_path": source_path,
                "source_count": source_count,
                "command": rebuild_command_for_profile(profile),
            }
        )

    needs_rebuild = any(report["status"] in {"missing", "stale", "unknown"} for report in reports)
    return not needs_rebuild, reports


def format_timestamp(value: float | None) -> str:
    if value is None:
        return "n/a"
    return datetime.fromtimestamp(value).isoformat(timespec="seconds")


def print_rebuild_advice() -> bool:
    fresh, reports = rebuild_advice()
    print("[Chronomere] rebuild advisor")
    print(f"[Chronomere] backup process: embedding-only rebuild with {EMBEDDING_ONLY_MODEL}; no LLM generation.")
    for report in reports:
        profile = str(report["profile"])
        status = str(report["status"])
        print(f"- {profile}: {status} ({report['reason']})")
        print(f"  index: {format_timestamp(report['index_mtime'])}")
        print(f"  newest source: {format_timestamp(report['source_mtime'])}")
        if report["source_path"] is not None:
            source_path = Path(str(report["source_path"]))
            try:
                source_display = source_path.relative_to(ROOT)
            except ValueError:
                source_display = source_path
            print(f"  source file: {source_display}")
        print(f"  source files scanned: {report['source_count']}")
        if report["missing"]:
            for path in report["missing"]:
                print(f"  missing: {path}")
        if status != "fresh":
            print("  rebuild: " + " ".join(str(part) for part in report["command"]))
    if fresh:
        print("[Chronomere] all active profile indexes look fresh.")
    else:
        print("[Chronomere] rebuild recommended for profiles marked missing, stale, or unknown.")
    return fresh


def context_budget_for_mode(args: argparse.Namespace, mode: str) -> Dict[str, int]:
    budget = dict(CONTEXT_BUDGETS[mode])
    overrides = {
        "top_k": args.top_k,
        "max_chunk_chars": args.max_chunk_chars,
        "num_predict": args.num_predict,
        "constraint_summary_limit": args.constraint_summary_limit,
    }

    for key, value in overrides.items():
        if value is not None:
            budget[key] = value

    if mode == "potato":
        budget["top_k"] = min(budget["top_k"], 2)
        budget["constraint_summary_limit"] = min(budget["constraint_summary_limit"], 2)

    return budget


def args_for_mode(args: argparse.Namespace, mode: str) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    copied.mode = mode
    return copied


def append_context_budget(command: List[str], budget: Dict[str, int]) -> None:
    command.extend(
        [
            "--top-k",
            str(budget["top_k"]),
            "--max-chunk-chars",
            str(budget["max_chunk_chars"]),
            "--num-predict",
            str(budget["num_predict"]),
            "--constraint-summary-limit",
            str(budget["constraint_summary_limit"]),
        ]
    )


def mode_uses_strict_citations(mode: str) -> bool:
    return mode in {"potato", "deep", "research"}


def should_audit_citations(args: argparse.Namespace, mode: str) -> bool:
    if args.stream:
        return False
    if args.audit_citations:
        return True
    return mode_uses_strict_citations(mode) and not args.no_auto_audit


def build_query_command(args: argparse.Namespace, mode: str) -> List[str]:
    question = args.question
    recalled_memory = getattr(args, "recalled_memory", [])
    if recalled_memory:
        question = memory_context_for_query(question, recalled_memory)

    command = [
        sys.executable,
        str(ROOT / "rag" / "query.py"),
        "--question",
        question,
        "--ollama-url",
        args.ollama_url,
    ]
    budget = context_budget_for_mode(args, mode)

    if mode == "potato":
        command.extend(["--profile", "quick", "--fast", "--strict-citations"])
    elif mode == "quick":
        command.extend(["--profile", "quick"])
    elif mode == "medium":
        command.extend(
            [
                "--profile",
                "medium",
                "--alpha-score",
                "--stability-rerank",
                "--fractal-context",
            ]
        )
    elif mode in {"deep", "research"}:
        command.extend(
            [
                "--profile",
                "deep",
                "--alpha-score",
                "--stability-rerank",
                "--fractal-context",
                "--strict-citations",
            ]
        )
    else:
        raise ValueError(f"Unknown Chronomere mode: {mode}")

    append_context_budget(command, budget)

    if args.show_context:
        command.append("--show-context")
    if args.stream:
        command.append("--stream")
    if args.print_timing:
        command.append("--print-timing")

    return command


def build_resolve_symbols_command(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(ROOT / "rag" / "symbol_resolver.py"),
        "--text",
        args.question,
        "--max-depth",
        str(args.symbol_depth),
    ]


def extract_answer_section(output: str) -> str:
    marker = "=== Answer ==="
    if marker not in output:
        return output

    answer = output.split(marker, 1)[1]
    for end_marker in ("\n=== Timing ===", "\n=== Citation Map ==="):
        if end_marker in answer:
            answer = answer.split(end_marker, 1)[0]
    return answer.strip()


def extract_citation_map_refs(output: str) -> Set[int]:
    marker = "=== Citation Map ==="
    if marker not in output:
        return set()

    citation_map = output.split(marker, 1)[1]
    return {int(match) for match in re.findall(r"^\[(\d+)\]", citation_map, flags=re.MULTILINE)}


def extract_answer_refs(answer: str) -> Set[int]:
    return {int(match) for match in re.findall(r"\[(\d+)\]", answer)}


def count_uncited_claim_lines(answer: str) -> int:
    count = 0
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("===", "#")):
            continue
        if re.fullmatch(r"\[\d+\].*", stripped):
            continue
        if re.search(r"\[\d+\]\s*$", stripped):
            continue
        count += 1
    return count


def print_citation_audit(output: str, strict_expected: bool) -> bool:
    answer = extract_answer_section(output)
    answer_refs = extract_answer_refs(answer)
    map_refs = extract_citation_map_refs(output)
    uncited_lines = count_uncited_claim_lines(answer)
    out_of_range = sorted(answer_refs - map_refs) if map_refs else []
    ok = bool(answer_refs) and not out_of_range
    if strict_expected and uncited_lines:
        ok = False

    print("\n=== Chronomere Citation Audit ===")
    if not answer_refs:
        print("- No answer citations found.")
    else:
        refs = ", ".join(f"[{ref}]" for ref in sorted(answer_refs))
        print(f"- Answer citations found: {refs}")

    if map_refs:
        print(f"- Citation map entries available: 1..{max(map_refs)}")
    else:
        print("- Citation map not found; range validation skipped.")

    if out_of_range:
        refs = ", ".join(f"[{ref}]" for ref in out_of_range)
        print(f"- Out-of-range citations: {refs}")

    if uncited_lines:
        print(f"- Lines without terminal citations: {uncited_lines}")
    else:
        print("- Every non-empty answer line ends with a citation or map entry.")

    print(f"- Audit result: {'pass' if ok else 'review'}")
    return ok


def answer_needs_escalation(output: str, return_code: int, audit_status: str) -> Tuple[bool, str]:
    if return_code != 0:
        return True, f"return_code={return_code}"
    if audit_status == "review":
        return True, "citation audit requested review"
    answer = extract_answer_section(output)
    if not extract_answer_refs(answer):
        return True, "no answer citations found"
    if not extract_citation_map_refs(output):
        return True, "citation map missing"
    return False, "answer passed mechanical checks"


def run_query(
    command: List[str],
    audit: bool,
    strict_expected: bool,
    passthrough: bool = False,
) -> Tuple[int, str, str]:
    if passthrough:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=subprocess_env(),
        )
        return completed.returncode, "skipped", ""

    if not audit:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=subprocess_env(),
        )
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)
        return completed.returncode, "skipped", completed.stdout

    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=subprocess_env(),
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)
    audit_status = "skipped"
    if completed.returncode == 0:
        audit_status = "pass" if print_citation_audit(completed.stdout, strict_expected) else "review"
    return completed.returncode, audit_status, completed.stdout


def write_session_log(record: Dict[str, object], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def build_session_record(
    args: argparse.Namespace,
    tool: str,
    mode: str,
    command: List[str],
    budget: Dict[str, int],
    audit_enabled: bool,
    audit_status: str,
    return_code: int,
    elapsed_seconds: float,
) -> Dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "question": args.question,
        "requested_mode": args.mode,
        "resolved_mode": mode,
        "dry_run": bool(args.dry_run),
        "stream": bool(args.stream),
        "show_context": bool(args.show_context),
        "print_timing": bool(args.print_timing),
        "index_check_skipped": bool(args.skip_index_check),
        "profiles": profiles_for_mode(mode),
        "budget": budget,
        "audit_enabled": audit_enabled,
        "audit_status": audit_status,
        "return_code": return_code,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "command": command,
        "memory_enabled": bool(getattr(args, "use_memory", False)),
        "memory_count": len(getattr(args, "recalled_memory", [])),
        "memory_ids": [
            str(record.get("id", "unknown"))
            for record in getattr(args, "recalled_memory", [])
        ],
        "tool_calls": [
            {
                "name": tool,
                "command": command,
                "return_code": return_code,
                "status": audit_status,
            }
        ],
    }


def build_escalation_session_record(
    args: argparse.Namespace,
    tool: str,
    attempts: List[Dict[str, object]],
    return_code: int,
    elapsed_seconds: float,
) -> Dict[str, object]:
    final_attempt = attempts[-1] if attempts else {}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "question": args.question,
        "requested_mode": args.mode,
        "resolved_mode": final_attempt.get("mode"),
        "dry_run": bool(args.dry_run),
        "stream": bool(args.stream),
        "show_context": bool(args.show_context),
        "print_timing": bool(args.print_timing),
        "index_check_skipped": bool(args.skip_index_check),
        "profiles": final_attempt.get("profiles", []),
        "budget": final_attempt.get("budget", {}),
        "audit_enabled": final_attempt.get("audit_enabled", False),
        "audit_status": final_attempt.get("audit_status", "skipped"),
        "return_code": return_code,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "command": final_attempt.get("command", []),
        "escalated": True,
        "attempt_count": len(attempts),
        "memory_enabled": bool(getattr(args, "use_memory", False)),
        "memory_count": len(getattr(args, "recalled_memory", [])),
        "memory_ids": [
            str(record.get("id", "unknown"))
            for record in getattr(args, "recalled_memory", [])
        ],
        "tool_calls": attempts,
    }


def build_tool_session_record(
    args: argparse.Namespace,
    tool: str,
    command: List[str],
    return_code: int,
    status: str,
    elapsed_seconds: float,
) -> Dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "question": args.question,
        "requested_mode": args.mode,
        "resolved_mode": None,
        "dry_run": bool(args.dry_run),
        "stream": False,
        "show_context": False,
        "print_timing": False,
        "index_check_skipped": None,
        "profiles": [],
        "budget": {},
        "audit_enabled": False,
        "audit_status": "not-applicable",
        "return_code": return_code,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "command": command,
        "memory_enabled": False,
        "memory_count": 0,
        "memory_ids": [],
        "tool_calls": [
            {
                "name": tool,
                "command": command,
                "return_code": return_code,
                "status": status,
            }
        ],
    }


def run_command_tool(
    args: argparse.Namespace,
    tool: str,
    command: List[str],
    started: float,
) -> int:
    print(f"[Chronomere] tool={tool}")
    print("[Chronomere] " + " ".join(command))

    if args.dry_run:
        if not args.no_session_log:
            record = build_tool_session_record(
                args,
                tool,
                command,
                0,
                "dry-run",
                time.perf_counter() - started,
            )
            write_session_log(record, Path(args.session_log))
            print(f"[Chronomere] session logged: {args.session_log}")
        return 0

    sys.stdout.flush()
    completed = subprocess.run(command, cwd=ROOT, env=subprocess_env())
    status = "pass" if completed.returncode == 0 else "error"
    if not args.no_session_log:
        record = build_tool_session_record(
            args,
            tool,
            command,
            completed.returncode,
            status,
            time.perf_counter() - started,
        )
        write_session_log(record, Path(args.session_log))
        print(f"[Chronomere] session logged: {args.session_log}")
    return completed.returncode


def run_remember_tool(args: argparse.Namespace, started: float) -> int:
    print("[Chronomere] tool=remember")
    if not args.memory_note:
        print("[Chronomere] --memory-note is required for --tool remember.")
        return 2

    if not preflight_before_tool_call(args, "remember", needs_ollama=bool(args.memory_vectorize)):
        return 2

    record = build_memory_record(args)
    memory_path = Path(args.memory_path)
    db_path = memory_db_path(args)
    status = "dry-run" if args.dry_run else "pass"
    if args.dry_run:
        print(f"[Chronomere] memory dry-run: {record['id']}")
    else:
        ensure_memory_db_seeded(memory_path, db_path)
        write_memory_record_sqlite(record, db_path)
        if args.memory_vectorize:
            vector = embed_memory_text(args.ollama_url, args.memory_embed_model, memory_text(record))
            write_memory_vector(db_path, str(record["id"]), args.memory_embed_model, vector)
        write_memory_record(record, memory_path)
        print(f"[Chronomere] memory saved: {record['id']}")
    print(f"[Chronomere] memory path: {memory_path}")
    print(f"[Chronomere] memory db: {db_path}")

    if not args.no_session_log:
        session = build_tool_session_record(
            args,
            "remember",
            ["chronomere", "remember", str(record["id"])],
            0,
            status,
            time.perf_counter() - started,
        )
        write_session_log(session, Path(args.session_log))
        print(f"[Chronomere] session logged: {args.session_log}")
    return 0


def run_recall_memory_tool(args: argparse.Namespace, started: float) -> int:
    print("[Chronomere] tool=recall-memory")
    if not preflight_before_tool_call(args, "recall-memory", needs_ollama=bool(args.memory_vector_recall)):
        return 2

    query = args.question or args.memory_note
    if not query:
        print("[Chronomere] --question or --memory-note is required for --tool recall-memory.")
        return 2

    memory_path = Path(args.memory_path)
    db_path = memory_db_path(args)
    imported = ensure_memory_db_seeded(memory_path, db_path)
    if imported:
        print(f"[Chronomere] seeded SQLite memory with {imported} JSONL atom(s).")
    records = recall_memory_records_hybrid(args, query)
    print(f"[Chronomere] memory db: {db_path}")
    print_memory_records(records)

    if not args.no_session_log:
        session = build_tool_session_record(
            args,
            "recall-memory",
            ["chronomere", "recall-memory", query],
            0,
            "pass",
            time.perf_counter() - started,
        )
        write_session_log(session, Path(args.session_log))
        print(f"[Chronomere] session logged: {args.session_log}")
    return 0


def run_escalation_loop(
    args: argparse.Namespace,
    tool: str,
    start_mode: str,
    started: float,
) -> int:
    if args.stream:
        print("[Chronomere] escalation loop does not support --stream yet.")
        return 2

    modes = escalation_plan(start_mode, args.max_escalation)
    attempts: List[Dict[str, object]] = []
    final_return_code = 0

    print(f"[Chronomere] tool={tool}")
    print("[Chronomere] escalation plan: " + " -> ".join(modes))

    for attempt_no, mode in enumerate(modes, start=1):
        attempt_args = args_for_mode(args, mode)
        attempt_args.audit_citations = True
        budget = context_budget_for_mode(attempt_args, mode)
        command = build_query_command(attempt_args, mode)
        profiles = profiles_for_mode(mode)
        audit = should_audit_citations(attempt_args, mode)

        print(f"[Chronomere] escalation attempt {attempt_no}/{len(modes)} mode={mode}")
        if not args.skip_index_check and not print_index_report(profiles):
            final_return_code = 2
            attempts.append(
                {
                    "name": f"{tool}:{mode}",
                    "mode": mode,
                    "profiles": profiles,
                    "budget": budget,
                    "audit_enabled": audit,
                    "audit_status": "index-check-failed",
                    "command": command,
                    "return_code": final_return_code,
                    "status": "index-check-failed",
                    "escalation_reason": "index check failed",
                }
            )
            break

        print(
            "[Chronomere] context budget: "
            f"top_k={budget['top_k']}, "
            f"max_chunk_chars={budget['max_chunk_chars']}, "
            f"num_predict={budget['num_predict']}, "
            f"constraint_summary_limit={budget['constraint_summary_limit']}"
        )
        print("[Chronomere] citation audit: enabled")
        print("[Chronomere] " + " ".join(command))

        if args.dry_run:
            attempts.append(
                {
                    "name": f"{tool}:{mode}",
                    "mode": mode,
                    "profiles": profiles,
                    "budget": budget,
                    "audit_enabled": audit,
                    "audit_status": "dry-run",
                    "command": command,
                    "return_code": 0,
                    "status": "dry-run",
                    "escalation_reason": "dry-run",
                }
            )
            continue

        sys.stdout.flush()
        return_code, audit_status, output = run_query(command, audit, mode_uses_strict_citations(mode))
        should_continue, reason = answer_needs_escalation(output, return_code, audit_status)
        final_return_code = return_code
        attempts.append(
            {
                "name": f"{tool}:{mode}",
                "mode": mode,
                "profiles": profiles,
                "budget": budget,
                "audit_enabled": audit,
                "audit_status": audit_status,
                "command": command,
                "return_code": return_code,
                "status": "escalated" if should_continue and attempt_no < len(modes) else "final",
                "escalation_reason": reason,
            }
        )

        if should_continue and attempt_no < len(modes):
            print(f"[Chronomere] escalating: {reason}")
            continue
        print(f"[Chronomere] escalation stopped: {reason}")
        break

    if args.dry_run:
        final_return_code = 0

    if not args.no_session_log:
        record = build_escalation_session_record(
            args,
            tool,
            attempts,
            final_return_code,
            time.perf_counter() - started,
        )
        write_session_log(record, Path(args.session_log))
        print(f"[Chronomere] session logged: {args.session_log}")
    return final_return_code


def main() -> int:
    configure_utf8_runtime()
    args = parse_args()
    started = time.perf_counter()
    tool = selected_tool(args)

    if args.list_tools:
        print_tool_registry()
        return 0

    if tool == "check-indexes":
        return 0 if print_index_report(PROFILE_INDEXES.keys()) else 2

    if tool == "rebuild-advice":
        return 0 if print_rebuild_advice() else 1

    if tool == "remember":
        return run_remember_tool(args, started)

    if tool == "recall-memory":
        return run_recall_memory_tool(args, started)

    if not args.question:
        print("[Chronomere] --question is required unless --tool check-indexes or --list-tools is used.")
        return 2

    if tool == "resolve-symbols":
        if not preflight_before_tool_call(args, tool, needs_ollama=False):
            return 2
        command = build_resolve_symbols_command(args)
        return run_command_tool(args, tool, command, started)

    if tool == "show-context":
        args.show_context = True
    if tool == "audit-citations":
        args.audit_citations = True

    mode = choose_mode(args.question) if args.mode == "auto" else args.mode
    if not preflight_before_tool_call(args, tool, needs_ollama=True):
        return 2
    args.recalled_memory = recalled_memory_for_args(args)
    if args.use_memory:
        print_memory_records(args.recalled_memory)
    if args.escalate:
        return run_escalation_loop(args, tool, mode, started)

    command = build_query_command(args, mode)
    audit = should_audit_citations(args, mode)
    budget = context_budget_for_mode(args, mode)

    print(f"[Chronomere] tool={tool}")
    print(f"[Chronomere] mode={mode}")
    if not args.skip_index_check and not print_index_report(profiles_for_mode(mode)):
        return 2
    print(
        "[Chronomere] context budget: "
        f"top_k={budget['top_k']}, "
        f"max_chunk_chars={budget['max_chunk_chars']}, "
        f"num_predict={budget['num_predict']}, "
        f"constraint_summary_limit={budget['constraint_summary_limit']}"
    )
    if args.stream and (args.audit_citations or mode_uses_strict_citations(mode)):
        print("[Chronomere] citation audit skipped during streaming output.")
    elif audit:
        print("[Chronomere] citation audit: enabled")
    print("[Chronomere] " + " ".join(command))

    if args.dry_run:
        if not args.no_session_log:
            record = build_session_record(
                args,
                tool,
                mode,
                command,
                budget,
                audit,
                "dry-run",
                0,
                time.perf_counter() - started,
            )
            write_session_log(record, Path(args.session_log))
            print(f"[Chronomere] session logged: {args.session_log}")
        return 0

    sys.stdout.flush()
    return_code, audit_status, _output = run_query(
        command,
        audit,
        mode_uses_strict_citations(mode),
        passthrough=args.stream,
    )
    if not args.no_session_log:
        record = build_session_record(
            args,
            tool,
            mode,
            command,
            budget,
            audit,
            audit_status,
            return_code,
            time.perf_counter() - started,
        )
        write_session_log(record, Path(args.session_log))
        print(f"[Chronomere] session logged: {args.session_log}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
