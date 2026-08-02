# TODO

## Enriched-question index: validate chunk_id alignment

The enriched questions (`enriched_questions.jsonl`) resolve to their source chunk
via `chunk_id`. Resolution assumes each question's `chunk_id` exists in the
*current* chunking produced by `chunk_corpus`.

Risk: the questions were generated against a specific chunking. If we later change
`--chunk-size` / `--chunk-overlap` (or the parsed corpus), chunk ids shift, those
questions silently fail to resolve, and `_resolve_to_chunks` just skips them
(the `resolved is None` branch in `rag_runner.py`). No error, no warning.

Action: add a startup check that logs how many enriched `chunk_id`s actually
match the current chunk ids, so a silent mismatch is visible. If we regenerate
chunks with different params, the question index must be regenerated too.
