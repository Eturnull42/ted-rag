#!/usr/bin/env python3
"""Build a lightweight local embedding index for TED-RAG."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import requests

TEXT_EXTENSIONS = {".txt", ".md", ".json", ".py", ".ps1"}
DEFAULT_EXCLUDE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".pdf", ".bin", ".pkl"}
DEFAULT_INCLUDE = ["rag/seed_context"]


@dataclass
class Chunk:
    source: str
    chunk_id: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build embedding index for TED-RAG")
    parser.add_argument("--root", default=".", help="Workspace root")
    parser.add_argument("--out", default="rag/index", help="Output index directory")
    parser.add_argument("--include", default=",".join(DEFAULT_INCLUDE), help="Comma-separated include paths")
    parser.add_argument(
        "--exclude-ext",
        default=",".join(sorted(DEFAULT_EXCLUDE_EXTENSIONS)),
        help="Comma-separated extension denylist (for example .jpg,.png,.pdf)",
    )
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=180, help="Chunk overlap in characters")
    parser.add_argument("--max-chunks", type=int, default=4000, help="Maximum chunks to embed in one run")
    parser.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    return parser.parse_args()


def iter_files(root: Path, include_paths: Iterable[str], exclude_extensions: set[str]) -> Iterator[Path]:
    for rel in include_paths:
        target = (root / rel.strip()).resolve()
        if not target.exists():
            continue
        suffix = target.suffix.lower()
        if target.is_file() and suffix in TEXT_EXTENSIONS and suffix not in exclude_extensions:
            yield target
            continue
        for path in target.rglob("*"):
            suffix = path.suffix.lower()
            if path.is_file() and suffix in TEXT_EXTENSIONS and suffix not in exclude_extensions:
                yield path


def flatten_json(obj: object, prefix: str = "") -> List[str]:
    lines: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(flatten_json(value, next_prefix))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            lines.extend(flatten_json(value, next_prefix))
    else:
        text = str(obj).strip()
        if text:
            if prefix:
                lines.append(f"{prefix}: {text}")
            else:
                lines.append(text)
    return lines


def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if suffix != ".json":
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return "\n".join(flatten_json(parsed))


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_math_aware(text: str, chunk_size: int, overlap: int) -> List[str]:
    # Prefer paragraph boundaries and keep equation/code blocks intact when possible.
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(para) <= chunk_size:
            current = para
            continue

        start = 0
        while start < len(para):
            end = min(start + chunk_size, len(para))
            window = para[start:end]

            split_idx = max(window.rfind("\n"), window.rfind(". "), window.rfind("; "))
            if split_idx > int(chunk_size * 0.6):
                end = start + split_idx + 1

            piece = para[start:end].strip()
            if piece:
                chunks.append(piece)

            if end >= len(para):
                break
            start = max(0, end - overlap)

        current = ""

    if current:
        chunks.append(current)

    return chunks


def embed_texts(base_url: str, model: str, texts: List[str]) -> np.ndarray:
    vectors: List[np.ndarray] = []
    endpoint = f"{base_url.rstrip('/')}/api/embeddings"

    for idx, text in enumerate(texts, start=1):
        response = requests.post(
            endpoint,
            json={"model": model, "prompt": text},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if not embedding:
            raise RuntimeError(f"Missing embedding for chunk {idx}")
        vectors.append(np.asarray(embedding, dtype=np.float32))

        if idx % 200 == 0:
            print(f"Embedded {idx}/{len(texts)} chunks...")

    matrix = np.vstack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    include_paths = [p.strip() for p in args.include.split(",") if p.strip()]
    exclude_extensions = {
        (ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}")
        for ext in args.exclude_ext.split(",")
        if ext.strip()
    }

    chunks: List[Chunk] = []
    file_count = 0

    for file_path in iter_files(root, include_paths, exclude_extensions):
        file_count += 1
        rel = file_path.relative_to(root).as_posix()
        text = normalize_text(read_text(file_path))
        if len(text) < 40:
            continue

        pieces = split_math_aware(text, args.chunk_size, args.chunk_overlap)
        for idx, piece in enumerate(pieces):
            chunks.append(Chunk(source=rel, chunk_id=idx, text=piece))
            if len(chunks) >= args.max_chunks:
                break
        if len(chunks) >= args.max_chunks:
            break

    if not chunks:
        raise RuntimeError("No chunks created. Check --include paths and file content.")

    print(f"Scanned {file_count} files; created {len(chunks)} chunks.")
    embeddings = embed_texts(args.ollama_url, args.embed_model, [c.text for c in chunks])

    np.save(out_dir / "embeddings.npy", embeddings)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(
                json.dumps(
                    {
                        "source": chunk.source,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest: Dict[str, object] = {
        "root": root.as_posix(),
        "include": include_paths,
        "exclude_extensions": sorted(exclude_extensions),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embed_model": args.embed_model,
        "chunk_count": len(chunks),
        "max_chunks": args.max_chunks,
        "embedding_dim": int(embeddings.shape[1]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Index written to: {out_dir}")


if __name__ == "__main__":
    main()
