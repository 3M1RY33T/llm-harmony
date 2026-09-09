# Fixtures

Recorded provider responses. Each says where it came from, because a fixture
that drifts from the real server is worse than no fixture.

| Fixture | Provenance |
|---|---|
| `lmstudio/models-none-loaded.json` | **Captured live** 2026-09-09, LM Studio on `:1234`, M4 Pro 24 GB |
| `lmstudio/models-one-loaded.json` | Derived from the above; `loaded_context_length: 8192` beside `max_context_length: 40960` per `docs/field-notes.md` |
| `ollama/ps-empty.json` | **Captured live** 2026-09-09, `GET /api/ps` with nothing resident |
| `ollama/tags.json` | **Captured live** 2026-09-09, `GET /api/tags` |
| `ollama/ps-one-loaded.json` | Derived from `tags.json` plus the `/api/ps` fields documented upstream |
| `llamacpp/v1-models.json` | **UNVERIFIED** — constructed from `docs/field-notes.md`; llama.cpp was not running |
| `llamacpp/running.json` | **UNVERIFIED** — same |
| `vllm/v1-models.json` | **UNVERIFIED** — constructed from `docs/field-notes.md`; vLLM-MLX was not running |

Re-capture the three UNVERIFIED fixtures the next time those servers run, then
update this table. If a re-capture disagrees with a fixture, the adapter is
wrong, not the server.
