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
| `llamacpp/v1-models.json` | **Captured live** 2026-09-09 from the router on `:8080` |
| `llamacpp/v1-models-one-loaded.json` | Derived from the above; `status.value` flipped to `loaded` |
| `vllm/v1-status.json` | **Captured live** 2026-09-10 from `/v1/status` on `:8000` |
| `vllm/v1-status-one-loaded.json` | Derived from the above; one entry's `loaded` flipped to true |

**No UNVERIFIED fixtures remain.** Every one above came from a live server.

The lesson cost two adapters. If a re-capture disagrees with a fixture, the
adapter is wrong, not the server.

**That is exactly what happened to llama.cpp on 2026-09-09.** The constructed
fixture claimed a `meta.n_ctx` block and a `/running` endpoint. The real router
has neither: state lives in `status.value`, `/running` returns 404, and no
context window is published at all. The adapter was rewritten to match the
server.
