#!/usr/bin/env python3
"""Run an example RAG query and write full output to rag/last_answer.txt."""
import subprocess, sys, pathlib

question = "Derive the key temporal equation and explain the assumptions behind it"
out_file = pathlib.Path("rag/last_answer.txt")

cmd = [
    sys.executable, "rag/query.py",
    "--question", question,
    "--index", "rag/index",
    "--top-k", "5",
    "--show-context",
    "--llm-model", "qwen2.5:3b-instruct-q4_K_M",
    "--embed-model", "nomic-embed-text",
    "--ollama-url", "http://127.0.0.1:11434",
]

result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
output = result.stdout + (("\n[STDERR]\n" + result.stderr) if result.stderr.strip() else "")
out_file.write_text(output, encoding="utf-8")
print(output)
print(f"\n[Written to {out_file}]")
