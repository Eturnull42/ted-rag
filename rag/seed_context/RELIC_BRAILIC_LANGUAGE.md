# Relic/Brailic Symbol Fallback

TED-RAG treats glyphic and Brailic forms as fallback representations for canonical symbols.

Rule:

1. Prefer the canonical symbol from `rag/symbol_table.json`.
2. If the canonical symbol is unavailable or cannot render, use the entry-specific Braille/Brailic/tactile representation.
3. If that is unavailable, use the ASCII alias.
4. Mark fallback representations as fallbacks rather than pretending they are canonical.

This lets symbolic retrieval degrade gracefully without emitting replacement characters such as `�`.
