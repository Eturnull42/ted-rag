import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from index_chronomere.conduct import (
    TOOL_REGISTRY,
    build_session_record,
    build_tool_session_record,
    build_resolve_symbols_command,
    context_budget_for_mode,
    answer_needs_escalation,
    escalation_plan,
    build_memory_record,
    memory_context_for_query,
    parse_memory_tags,
    read_memory_records,
    read_memory_records_sqlite,
    recall_memory_records_hybrid,
    recall_memory_records,
    read_memory_vectors,
    subprocess_env,
    validate_numeric_args,
    write_memory_record,
    write_memory_record_sqlite,
    write_memory_vector,
    ensure_memory_db_seeded,
    index_timestamp,
    latest_source_mtime,
    rebuild_command_for_profile,
    extract_answer_refs,
    extract_answer_section,
    extract_citation_map_refs,
    count_uncited_claim_lines,
    write_session_log,
)


class ChronomereCitationAuditTests(unittest.TestCase):
    def test_extracts_answer_refs_and_map_refs(self) -> None:
        output = """=== Answer ===
- First claim [1]
- Second claim [2]

=== Citation Map ===
[1] a.md (chunk 0)
[2] b.md (chunk 1)
"""
        answer = extract_answer_section(output)

        self.assertEqual(extract_answer_refs(answer), {1, 2})
        self.assertEqual(extract_citation_map_refs(output), {1, 2})
        self.assertEqual(count_uncited_claim_lines(answer), 0)

    def test_counts_uncited_answer_lines(self) -> None:
        answer = "- Missing citation\n- Has citation [1]"

        self.assertEqual(count_uncited_claim_lines(answer), 1)

    def test_writes_session_log_jsonl(self) -> None:
        args = SimpleNamespace(
            question="Define S_ent",
            mode="auto",
            dry_run=True,
            stream=False,
            show_context=False,
            print_timing=False,
            skip_index_check=False,
            top_k=None,
            max_chunk_chars=None,
            num_predict=None,
            constraint_summary_limit=None,
        )
        budget = context_budget_for_mode(args, "quick")
        record = build_session_record(
            args,
            "query",
            "quick",
            ["py", "rag/query.py"],
            budget,
            audit_enabled=False,
            audit_status="dry-run",
            return_code=0,
            elapsed_seconds=0.25,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "chronomere_sessions.jsonl"
            write_session_log(record, log_path)
            stored = json.loads(log_path.read_text(encoding="utf-8").strip())

        self.assertEqual(stored["resolved_mode"], "quick")
        self.assertEqual(stored["tool"], "query")
        self.assertTrue(stored["dry_run"])
        self.assertEqual(stored["budget"]["top_k"], 4)
        self.assertEqual(stored["return_code"], 0)
        self.assertEqual(stored["tool_calls"][0]["name"], "query")

    def test_tool_registry_includes_core_tools(self) -> None:
        self.assertIn("query", TOOL_REGISTRY)
        self.assertIn("check-indexes", TOOL_REGISTRY)
        self.assertIn("rebuild-advice", TOOL_REGISTRY)
        self.assertIn("resolve-symbols", TOOL_REGISTRY)
        self.assertIn("audit-citations", TOOL_REGISTRY)

    def test_subprocess_env_forces_utf8(self) -> None:
        env = subprocess_env()

        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(env["PYTHONUTF8"], "1")

    def test_builds_resolve_symbols_command(self) -> None:
        args = SimpleNamespace(question="Resolve S_ent", symbol_depth=2)
        command = build_resolve_symbols_command(args)

        self.assertIn("symbol_resolver.py", command[1])
        self.assertIn("--text", command)
        self.assertIn("Resolve S_ent", command)
        self.assertIn("--max-depth", command)
        self.assertIn("2", command)

    def test_writes_tool_session_record(self) -> None:
        args = SimpleNamespace(
            question="Resolve S_ent",
            mode="auto",
            dry_run=True,
        )
        record = build_tool_session_record(
            args,
            "resolve-symbols",
            ["py", "rag/symbol_resolver.py"],
            return_code=0,
            status="dry-run",
            elapsed_seconds=0.1,
        )

        self.assertIsNone(record["resolved_mode"])
        self.assertEqual(record["tool"], "resolve-symbols")
        self.assertEqual(record["tool_calls"][0]["status"], "dry-run")
        self.assertEqual(record["tool_calls"][0]["name"], "resolve-symbols")

    def test_latest_source_mtime_scans_default_include_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            seed_dir = root / "rag" / "seed_context"
            seed_dir.mkdir(parents=True)
            source = seed_dir / "TED_RAG_ALPHA.md"
            source.write_text("Temporal entanglement context", encoding="utf-8")
            os.utime(source, (1000, 1000))

            mtime, path, count = latest_source_mtime(root)

        self.assertEqual(mtime, 1000)
        self.assertEqual(path, source)
        self.assertEqual(count, 1)

    def test_index_timestamp_reports_missing_and_present_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_dir = Path(tmp_dir)
            embeddings = index_dir / "embeddings.npy"
            chunks = index_dir / "chunks.jsonl"
            embeddings.write_text("embeddings", encoding="utf-8")
            os.utime(embeddings, (2000, 2000))

            missing_time, missing = index_timestamp(index_dir)
            chunks.write_text("chunks", encoding="utf-8")
            os.utime(chunks, (1500, 1500))
            present_time, present_missing = index_timestamp(index_dir)

        self.assertIsNone(missing_time)
        self.assertTrue(any(path.name == "chunks.jsonl" for path in missing))
        self.assertEqual(present_time, 1500)
        self.assertEqual(present_missing, [])

    def test_rebuild_command_uses_profile_limits(self) -> None:
        command = rebuild_command_for_profile("medium")

        self.assertIn("-u", command)
        self.assertTrue(any("build_index.py" in part for part in command))
        self.assertIn("rag/index_medium", command)
        self.assertIn("--max-chunks", command)
        self.assertIn("500", command)
        self.assertIn("--embed-model", command)
        self.assertIn("nomic-embed-text", command)

    def test_escalation_plan_climbs_to_max_mode(self) -> None:
        self.assertEqual(escalation_plan("quick", "research"), ["quick", "medium", "research"])
        self.assertEqual(escalation_plan("medium", "research"), ["medium", "research"])
        self.assertEqual(escalation_plan("research", "medium"), ["research"])

    def test_answer_needs_escalation_for_missing_citations(self) -> None:
        output = """=== Answer ===
No citation here.

=== Citation Map ===
[1] a.md (chunk 0)
"""
        should_escalate, reason = answer_needs_escalation(output, 0, "skipped")

        self.assertTrue(should_escalate)
        self.assertIn("no answer citations", reason)

    def test_answer_passes_escalation_checks_with_citations_and_map(self) -> None:
        output = """=== Answer ===
- Claim [1]

=== Citation Map ===
[1] a.md (chunk 0)
"""
        should_escalate, reason = answer_needs_escalation(output, 0, "pass")

        self.assertFalse(should_escalate)
        self.assertIn("passed", reason)

    def test_validate_numeric_args_flags_bad_values(self) -> None:
        args = SimpleNamespace(
            top_k=0,
            max_chunk_chars=-1,
            num_predict=1,
            constraint_summary_limit=1,
            symbol_depth=-1,
            memory_limit=0,
        )
        problems = validate_numeric_args(args)

        self.assertTrue(any("--top-k" in problem for problem in problems))
        self.assertTrue(any("--max-chunk-chars" in problem for problem in problems))
        self.assertTrue(any("--symbol-depth" in problem for problem in problems))
        self.assertTrue(any("--memory-limit" in problem for problem in problems))

    def test_memory_tags_are_normalized(self) -> None:
        tags = parse_memory_tags(" ILE, Index Chronomere, ile ")

        self.assertEqual(tags, ["ile", "index-chronomere"])

    def test_writes_and_recalls_memory_record(self) -> None:
        args = SimpleNamespace(
            memory_note="ILE lives in Chronomere memory as curated continuity.",
            memory_title="ILE Memory Layer",
            memory_kind="lore",
            memory_tags="ile, chronomere, memory",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_path = Path(tmp_dir) / "memory.jsonl"
            record = build_memory_record(args)
            write_memory_record(record, memory_path)

            stored = read_memory_records(memory_path)
            recalled = recall_memory_records("chronomere ile continuity", memory_path, 3)

        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["title"], "ILE Memory Layer")
        self.assertEqual(len(recalled), 1)
        self.assertEqual(recalled[0]["title"], "ILE Memory Layer")

    def test_seeds_sqlite_memory_from_jsonl(self) -> None:
        args = SimpleNamespace(
            memory_note="ILE remains the Chronomere core.",
            memory_title="ILE Core",
            memory_kind="lore",
            memory_tags="ile, core",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_path = Path(tmp_dir) / "memory.jsonl"
            db_path = Path(tmp_dir) / "memory.sqlite"
            write_memory_record(build_memory_record(args), memory_path)

            imported = ensure_memory_db_seeded(memory_path, db_path)
            stored = read_memory_records_sqlite(db_path)

        self.assertEqual(imported, 1)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["title"], "ILE Core")

    def test_hybrid_recall_uses_sqlite_lexical_without_vectors(self) -> None:
        args = SimpleNamespace(
            memory_path="",
            memory_db="",
            memory_limit=3,
            memory_vector_recall=False,
        )
        record_args = SimpleNamespace(
            memory_note="Chronomere memory scales as a curated soul-thread.",
            memory_title="Soul Thread",
            memory_kind="architecture",
            memory_tags="chronomere, memory",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            args.memory_path = str(Path(tmp_dir) / "memory.jsonl")
            args.memory_db = str(Path(tmp_dir) / "memory.sqlite")
            write_memory_record_sqlite(build_memory_record(record_args), Path(args.memory_db))
            recalled = recall_memory_records_hybrid(args, "curated chronomere soul thread")

        self.assertEqual(len(recalled), 1)
        self.assertEqual(recalled[0]["title"], "Soul Thread")

    def test_writes_and_reads_memory_vector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "memory.sqlite"
            record = {
                "id": "ile-core",
                "timestamp_utc": "2026-06-30T00:00:00+00:00",
                "kind": "lore",
                "title": "ILE Core",
                "summary": "ILE remains core.",
                "note": "ILE remains core.",
                "tags": ["ile"],
            }
            write_memory_record_sqlite(record, db_path)
            write_memory_vector(db_path, "ile-core", "test-embed", np.asarray([1.0, 0.0], dtype=np.float32))
            vectors = read_memory_vectors(db_path, "test-embed")

        self.assertIn("ile-core", vectors)
        self.assertEqual(vectors["ile-core"].shape[0], 2)

    def test_memory_context_wraps_question(self) -> None:
        context = memory_context_for_query(
            "Explain Index's magical structure",
            [
                {
                    "title": "ILE Memory Layer",
                    "summary": "ILE lives here as Chronomere's curated memory.",
                    "tags": ["ile", "memory"],
                }
            ],
        )

        self.assertIn("Chronomere memory context:", context)
        self.assertIn("ILE Memory Layer", context)
        self.assertIn("Current question:", context)


if __name__ == "__main__":
    unittest.main()
