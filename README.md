# Personalized News Headlines with RAG

This repository preserves the original PENS/CFRAG experiment code used by the notebooks `Untitled6.ipynb` and `Untitled10.ipynb`. It does not replace the project with a newly designed Python package.

## What is original

The Python files under `rag/` were copied from the Google Drive folder `LaMP_checkpoints/CFRAG` by following the direct and transitive imports of the commands used in `Untitled10`:

- `ranking.py` and its retrieval/reranking dependencies;
- `generation/generate.py`;
- `data/datasets.py` and `data/preprocess_profile.py`;
- the original prompt, post-processing, and metric files.

The merged notebook contains 424 source lines copied from the two original notebooks and 76 lines of repository glue. Every copied cell has `metadata.repo_provenance` recording its source notebook and cell number.

## Four deliberate source changes

Only three Python changes were made:

1. `prompts/prompts.py`: re-enabled the existing, commented prompt block that concatenates `history`. This matches the prompt printed in the saved `Untitled10` output.
2. `runners/RetrievalRunner.py`: `bm25_profile` skips `load_user()`, so this paper path does not require trained user vectors or collaborative filtering.
3. `ranking.py`: added `Llama-3.1-8B-Instruct` as the default output-directory model name.
4. `generation/generate.py`: added `type=int` to the two existing length arguments so parameterized notebook values are not passed to vLLM as strings.

See [docs/code-trace.md](docs/code-trace.md) for the exact call graph and [docs/original-code-manifest.md](docs/original-code-manifest.md) for Drive IDs.

## Run without Google Drive

Clone or unpack the repository, then place the PENS files locally:

```text
data/raw/news.tsv
data/raw/personalized_test.tsv
```

Alternatively, set the `PENS_DATA_ROOT` environment variable to the directory containing those two files. Start Jupyter from the repository root (or open the notebook from `notebooks/`) and run `notebooks/pens_rag_pipeline.ipynb`.

The notebook uses `Path.cwd()`, `sys.executable`, and explicit subprocess working directories. It contains no Google Drive mount, no `/content/drive` paths, and no Drive symlink.

It runs:

1. local dependency installation;
2. original-headline and rewritten-headline preprocessing;
3. one selected original retrieval branch: BM25, BGE, or random;
4. prompt inspection and a hard 128k context check;
5. the original `generation/generate.py` pipeline through a portable subprocess command.

Change only the configuration cell:

```python
TARGET = "rewritten"       # original | rewritten
RETRIEVER = "bm25"         # bm25 | dense | random
TOP_K = 10
CONTEXT_LENGTH = 128_000
BEGIN_INDEX = 0
END_INDEX = 20_600
```

The context audit uses the original `Seq2SeqDataset` and prompt function. It stops before vLLM if the complete Top-k prompt plus 64 output tokens exceeds the selected limit; it does not silently truncate the history.

Preprocessing and retrieval can run in a normal Python/Jupyter environment. vLLM generation requires a compatible Linux CUDA environment and access to the selected Hugging Face model.

## Repository layout

```text
rag/                                Original CFRAG Python call graph
notebooks/pens_rag_pipeline.ipynb   Original-cell merged Colab notebook
docs/code-trace.md                  Call graph and exclusions
docs/original-code-manifest.md      Google Drive source IDs and patch ledger
```

PENS data, model weights, generated retrieval files, and model outputs are intentionally excluded from Git.
