from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

RAG_DIR = Path(__file__).resolve().parents[1]
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from context_builder import build_fractal_constraint
from query import (
    build_prompt,
    format_symbol_validation,
    inject_missing_citations,
    streaming_potato_controls,
    validate_symbol_dependencies,
)
from scoring import (
    alpha_score,
    estimate_fractal_dimension,
    stability_rerank,
    unresolved_symbol_count,
)
from symbol_resolver import extract_symbols, load_symbol_table, resolve_dependencies


TABLE_PATH = RAG_DIR / "symbol_table.json"


class TedRagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.table = load_symbol_table(TABLE_PATH)

    def test_symbol_resolution_finds_dependencies(self) -> None:
        matches = extract_symbols("Use S_ent and Psi", self.table)
        roots = [match.symbol for match in matches]
        resolution = resolve_dependencies(roots, self.table, max_depth=1)

        self.assertIn("S_ent", roots)
        self.assertIn("Ψ", roots)
        self.assertIn("D[Ψ']", resolution["symbols"])
        self.assertIn("S_tachyon", resolution["symbols"])

    def test_dependency_validation_reports_missing_context(self) -> None:
        ranked = [(1.0, {"source": "fake.md", "chunk_id": 0, "text": "Only S_ent appears here."})]
        validation = validate_symbol_dependencies("Use S_ent and Psi", ranked, TABLE_PATH, 1)
        formatted = format_symbol_validation(validation)

        self.assertIn("S_tachyon", validation["missing_from_context"])
        self.assertIn("Representation fallback", formatted)

    def test_alpha_score_supports_risk_and_novelty_weights(self) -> None:
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        chunk_vec = np.array([1.0, 0.0], dtype=np.float32)
        corpus = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        text = "tachyon instability with S_tachyon"

        risk_only = alpha_score(query_vec, chunk_vec, corpus, text, self.table, lambda_risk=0.2)
        novelty_only = alpha_score(
            query_vec,
            chunk_vec,
            corpus,
            text,
            self.table,
            lambda_risk=0.0,
            lambda_novelty=0.2,
        )

        self.assertGreater(risk_only.tachyon_risk, 0.0)
        self.assertEqual(risk_only.tachyon_risk, risk_only.tachyon_novelty)
        self.assertGreater(novelty_only.total, risk_only.total)

    def test_stability_rerank_computes_eta_gamma(self) -> None:
        plain = "This is ordinary prose about retrieval."
        mathy = "S_ent[Psi] = lambda * S_tachyon\n| term | value |\n| A | B |"
        noisy = "Omega_unknown[42] couples to Ξ_hidden."

        self.assertGreater(estimate_fractal_dimension(mathy), estimate_fractal_dimension(plain))
        self.assertGreater(unresolved_symbol_count(noisy, self.table), 0)

        reranked = stability_rerank(
            [
                (1.0, {"source": "plain", "chunk_id": 0, "text": plain}),
                (1.0, {"source": "mathy", "chunk_id": 0, "text": mathy}),
            ],
            self.table,
        )
        self.assertTrue(all("stability_score" in chunk for _, chunk in reranked))

    def test_fractal_constraint_weights_sum_to_one(self) -> None:
        ranked = [
            (2.0, {"source": "a.md", "chunk_id": 0, "text": "A", "ted_role": "definition"}),
            (1.0, {"source": "b.md", "chunk_id": 1, "text": "B"}),
        ]
        weighted, summary = build_fractal_constraint(ranked)

        self.assertAlmostEqual(summary["weight_sum"], 1.0)
        self.assertAlmostEqual(sum(chunk["constraint"]["weight"] for _, chunk in weighted), 1.0)
        self.assertEqual(weighted[0][1]["constraint"]["term"], "C_1")

    def test_citation_injection_adds_missing_refs(self) -> None:
        ranked = [(1.0, {"source": "a.md", "chunk_id": 0, "text": "A"})]
        fixed, changed = inject_missing_citations("- Missing citation\n- Already cited [1]", ranked)

        self.assertTrue(changed)
        self.assertIn("- Missing citation [1]", fixed)
        self.assertIn("- Already cited [1]", fixed)

    def test_strict_citation_prompt_contains_rules(self) -> None:
        ranked = [(1.0, {"source": "a.md", "chunk_id": 0, "text": "Context text"})]
        prompt = build_prompt("Question?", ranked, strict_citations=True)

        self.assertIn("Every sentence must end with", prompt)
        self.assertIn("Use Ψ' not Φ'", prompt)
        self.assertIn("Answer format:", prompt)

    def test_streaming_potato_controls_apply_fast_defaults(self) -> None:
        args = type(
            "Args",
            (),
            {
                "llm_model": "larger-model",
                "top_k": 8,
                "max_chunk_chars": None,
                "num_predict": None,
                "temperature": None,
                "alpha_score": False,
                "alpha_candidates": 64,
                "stability_rerank": False,
                "fractal_context": False,
                "constraint_summary_limit": 12,
            },
        )()

        tuned = streaming_potato_controls(args)

        self.assertEqual(tuned.llm_model, "qwen2.5:1.5b")
        self.assertEqual(tuned.top_k, 2)
        self.assertEqual(tuned.max_chunk_chars, 600)
        self.assertEqual(tuned.num_predict, 120)
        self.assertEqual(tuned.temperature, 0.0)
        self.assertTrue(tuned.alpha_score)
        self.assertTrue(tuned.stability_rerank)
        self.assertTrue(tuned.fractal_context)


if __name__ == "__main__":
    unittest.main()
