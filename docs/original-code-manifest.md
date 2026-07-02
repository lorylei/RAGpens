# Original-code manifest

These files were fetched as raw bytes from Google Drive on 2026-07-02.

| Repository path | Google Drive file ID | Status |
|---|---|---|
| `rag/ranking.py` | `1JdmcaiRfhIRLieOrkLxX0uDRHS6CVnBU` | Original plus documented model-name edit |
| `rag/runners/RetrievalRunner.py` | `1_RDCVfsbA5ww45Sc5cgDq_uIH0hf7yXM` | Original plus documented BM25 user-vector bypass |
| `rag/runners/ReRankRunner.py` | `1aZ8P00ZQZQHMENYXoWPWWDEm901XSTBt` | Copied without logical edits (LF line endings) |
| `rag/models/retriever.py` | `1pz66Lz3uX9x6lTsx-o91zTSqgtOfpUZK` | Copied without logical edits (LF line endings) |
| `rag/models/reranker.py` | `1aOmF3DD5g7Xex2Kd1S5PxoBfZzWB_D8p` | Copied without logical edits (LF line endings) |
| `rag/generation/generate.py` | `1QoIWjBk4POXkQ-flLK0N329rv8zrgrqt` | Original plus documented argparse type edit |
| `rag/data/datasets.py` | `19aSfDhBGYmh4QSiur5v5azTylW08oi7O` | Copied without logical edits (LF line endings) |
| `rag/data/preprocess_profile.py` | `16X55Fl0JXhX73bD9CHd2NDY8zhAte4ZD` | Copied without logical edits (LF line endings) |
| `rag/metrics/eval_metrics.py` | `1Ak9IGF8tMTDnMbngKw2THsQOZlJKjib0` | Copied without logical edits (LF line endings) |
| `rag/prompts/post_process.py` | `1B3wjmE8YbH2iO1pVohgirM5v_mOGUdd0` | Copied without logical edits (LF line endings) |
| `rag/prompts/pre_process.py` | `1t_eyrOQB8EZ97CaBctFGxnkDy0_eheHG` | Copied without logical edits (LF line endings) |
| `rag/prompts/prompts.py` | `1qrRv-d9gSR282Xoj1YwQFLjK7qAVmI_6` | Original plus re-enabled existing history block |
| `rag/requirements.txt` | `1K5jNy9yeR0qaOQukk0Jq7ca3maY8nyQK` | Copied without logical edits (LF line endings) |

## Patch ledger

No Python module was newly designed for this repository.

1. `prompts/prompts.py`: uncommented the existing prompt block that includes `history`, matching the prompt preserved in `Untitled10` outputs.
2. `runners/RetrievalRunner.py`: moved `load_user(opts)` into the non-`bm25_profile` branch. This prevents the paper's BM25 path from requiring CFRAG user embeddings.
3. `ranking.py`: added `Llama-3.1-8B-Instruct` to the existing output-folder choices and made it the default.
4. `generation/generate.py`: added `type=int` to `max_new_tokens` and `cutoff_len`, because argparse otherwise converts explicitly supplied notebook values to strings.
