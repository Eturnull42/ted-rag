#!/usr/bin/env python3
"""Resolve TED-RAG symbols and dependencies from the symbol table."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TABLE = Path(__file__).with_name("symbol_table.json")


@dataclass(frozen=True)
class SymbolMatch:
    symbol: str
    matched: str
    start: int
    end: int


def load_symbol_table(path: Path = DEFAULT_TABLE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict) or not isinstance(data.get("symbols"), dict):
        raise ValueError(f"{path} does not look like a TED-RAG symbol table")

    validate_symbol_table(data)
    return data


def validate_symbol_table(table: dict[str, Any]) -> None:
    symbols = table["symbols"]
    required_fields = {"kind", "defined_in", "depends_on", "use"}

    errors: list[str] = []
    for name, entry in symbols.items():
        if not isinstance(entry, dict):
            errors.append(f"{name}: entry must be an object")
            continue

        missing = sorted(required_fields.difference(entry))
        if missing:
            errors.append(f"{name}: missing fields {', '.join(missing)}")

        if "defined_in" in entry and not isinstance(entry["defined_in"], list):
            errors.append(f"{name}: defined_in must be a list")

        if "depends_on" in entry and not isinstance(entry["depends_on"], list):
            errors.append(f"{name}: depends_on must be a list")

        if "aliases" in entry and not isinstance(entry["aliases"], list):
            errors.append(f"{name}: aliases must be a list")

    dependencies = {
        dep
        for entry in symbols.values()
        if isinstance(entry, dict)
        for dep in entry.get("depends_on", [])
    }
    undefined = sorted(dep for dep in dependencies if dep not in symbols)
    if undefined:
        errors.append("undefined dependencies: " + ", ".join(undefined))

    if errors:
        raise ValueError("Invalid symbol table:\n- " + "\n- ".join(errors))


def representation_for_symbol(symbol: str, table: dict[str, Any]) -> dict[str, str]:
    """Return the best display fallback for a symbol under table policy."""
    entry = table["symbols"].get(symbol)
    if not isinstance(entry, dict):
        return {"symbol": symbol, "representation": symbol, "source": "missing_symbol"}

    policy = table.get("policy", {}).get("representation_fallback", {})
    fallback_fields = policy.get("fallback_fields", [])
    if not isinstance(fallback_fields, list):
        fallback_fields = []

    for field in fallback_fields:
        value = entry.get(field)
        if isinstance(value, str) and value.strip():
            return {
                "symbol": symbol,
                "representation": value.strip(),
                "source": str(field),
            }

    alias_terms = policy.get("fallback_alias_terms", [])
    if not isinstance(alias_terms, list):
        alias_terms = []
    alias_terms = [str(term).lower() for term in alias_terms]

    aliases = entry.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []

    for alias in aliases:
        if not isinstance(alias, str) or not alias.strip():
            continue
        lowered = alias.lower()
        if any(term in lowered for term in alias_terms):
            return {
                "symbol": symbol,
                "representation": alias.strip(),
                "source": "semantic_alias",
            }

    if policy.get("ascii_alias_fallback", True):
        for alias in aliases:
            if isinstance(alias, str) and alias.strip() and alias.isascii():
                return {
                    "symbol": symbol,
                    "representation": alias.strip(),
                    "source": "ascii_alias",
                }

    return {"symbol": symbol, "representation": symbol, "source": "canonical"}


def fallback_representations(symbols: Iterable[str], table: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {symbol: representation_for_symbol(symbol, table) for symbol in dict.fromkeys(symbols)}


def iter_terms(symbols: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for symbol, entry in symbols.items():
        yield symbol, symbol
        for alias in entry.get("aliases", []):
            if isinstance(alias, str) and alias:
                yield alias, symbol


def is_wordlike(term: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_ -]+", term))


def term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    if is_wordlike(term):
        return re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", re.IGNORECASE)
    return re.compile(escaped)


def extract_symbols(text: str, table: dict[str, Any]) -> list[SymbolMatch]:
    symbols = table["symbols"]
    matches: list[SymbolMatch] = []

    terms = sorted(iter_terms(symbols), key=lambda item: len(item[0]), reverse=True)
    occupied: list[tuple[int, int]] = []

    for term, symbol in terms:
        for match in term_pattern(term).finditer(text):
            span = match.span()
            if any(not (span[1] <= start or span[0] >= end) for start, end in occupied):
                continue
            occupied.append(span)
            matches.append(SymbolMatch(symbol=symbol, matched=match.group(0), start=span[0], end=span[1]))

    matches.sort(key=lambda item: (item.start, item.end, item.symbol))
    return matches


def resolve_dependencies(
    symbols: Iterable[str],
    table: dict[str, Any],
    max_depth: int | None = None,
) -> dict[str, Any]:
    entries = table["symbols"]
    roots = list(dict.fromkeys(symbols))
    resolved: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    missing: list[str] = []

    def visit(symbol: str, depth: int, stack: tuple[str, ...]) -> None:
        if max_depth is not None and depth > max_depth:
            return

        if symbol not in entries:
            if symbol not in missing:
                missing.append(symbol)
            return

        if symbol not in resolved:
            entry = entries[symbol]
            resolved[symbol] = {
                "kind": entry.get("kind"),
                "defined_in": entry.get("defined_in", []),
                "depends_on": entry.get("depends_on", []),
                "formula": entry.get("formula"),
                "use": entry.get("use"),
            }

        if symbol in stack:
            return

        if max_depth is not None and depth >= max_depth:
            return

        for dep in entries[symbol].get("depends_on", []):
            edges.append({"from": symbol, "to": dep})
            visit(dep, depth + 1, (*stack, symbol))

    for root in roots:
        visit(root, 0, ())

    return {
        "roots": roots,
        "symbols": resolved,
        "edges": edges,
        "missing": missing,
    }


def resolution_for_text(text: str, table: dict[str, Any], max_depth: int | None = None) -> dict[str, Any]:
    matches = extract_symbols(text, table)
    roots = [match.symbol for match in matches]
    resolution = resolve_dependencies(roots, table, max_depth=max_depth)
    resolution["matches"] = [
        {
            "symbol": match.symbol,
            "matched": match.matched,
            "start": match.start,
            "end": match.end,
        }
        for match in matches
    ]
    resolution["representations"] = fallback_representations(resolution["symbols"], table)
    return resolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve TED-RAG symbols and dependencies")
    parser.add_argument("--table", default=str(DEFAULT_TABLE), help="Path to symbol_table.json")
    parser.add_argument("--text", help="Text to scan for symbols")
    parser.add_argument("--file", help="File to scan for symbols")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum dependency depth")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--validate-only", action="store_true", help="Only validate the symbol table")
    return parser.parse_args()


def read_input(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        return Path(args.file).read_text(encoding="utf-8")
    raise SystemExit("Provide --text, --file, or --validate-only")


def print_human(result: dict[str, Any]) -> None:
    print("Matches:")
    for match in result["matches"]:
        print(f"- {match['symbol']} via {match['matched']!r} at {match['start']}:{match['end']}")

    print("\nResolved symbols:")
    for symbol, entry in result["symbols"].items():
        deps = ", ".join(entry.get("depends_on", [])) or "(none)"
        representation = result.get("representations", {}).get(symbol, {})
        fallback = representation.get("representation", symbol)
        source = representation.get("source", "canonical")
        print(f"- {symbol} [{entry.get('kind')}]: depends_on={deps}; fallback={fallback} ({source})")

    if result["missing"]:
        print("\nMissing dependencies:")
        for symbol in result["missing"]:
            print(f"- {symbol}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    table = load_symbol_table(Path(args.table))

    if args.validate_only:
        print(f"OK: {args.table}")
        return

    result = resolution_for_text(read_input(args), table, max_depth=args.max_depth)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_human(result)


if __name__ == "__main__":
    main()
