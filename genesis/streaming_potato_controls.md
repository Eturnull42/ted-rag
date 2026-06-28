# Streaming Potato Controls

`streaming_potato_controls()` is the CPU-friendly preset for TED-RAG.

Defaults:

- `llm_model = qwen2.5:1.5b`
- `top_k = 2`
- `max_chunk_chars = 600`
- `num_predict = 120`
- `temperature = 0`
- Alpha scoring enabled
- Stability rerank enabled
- Fractal context enabled
- Constraint summary limited to two terms

The principle is to run expensive stability logic only on contenders, keep prompts small, and stream generation as soon as retrieval is complete.
