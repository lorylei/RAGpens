# Untitled10 call-graph audit

## Boundary

`Untitled10.ipynb` is the only root used to decide which CFRAG Python files belong in the repository. A file is included when it is directly invoked by a retained notebook command or imported transitively by that file.

## Retained generation chain

```text
generation/generate.py
├── data/datasets.py
│   ├── prompts/pre_process.py
│   └── prompts/prompts.py
│       └── prompts/pre_process.py
├── metrics/eval_metrics.py
└── prompts/post_process.py
```

## Retained ranking chain

```text
ranking.py
├── runners/RetrievalRunner.py
│   ├── models/retriever.py
│   └── prompts/pre_process.py
└── runners/ReRankRunner.py
    ├── models/reranker.py
    └── prompts/pre_process.py
```

The reranking files remain because the original `ranking.py` imports them at module import time. The merged paper workflow calls only `--rank_stage retrieval --ret_type bm25_profile`.

Some original classes still contain CFRAG's dense/user-embedding branches because the user requested the original Python files rather than rewritten replacements. They are not executed by the retained notebook path. The only behavioral patch is that `bm25_profile` no longer calls `load_user()`.

## Notebook cells retained

- `Untitled6` cells 11 and 12: test-set preprocessing with original and rewritten targets.
- `Untitled10` cells 5 and 7: the original dense and random retrieval implementations.
- The BM25 command is the command from `Untitled10` cell 32 with local paths, k, and shard bounds exposed as variables.
- The generation subprocess is the command from `Untitled10` cell 48, expressed with `sys.executable` and an explicit local working directory.

Drive mounting cells 53/54 and all `/content/drive` paths are excluded from the portable notebook.

Outputs and execution counters are removed, but copied source bodies are preserved. Mechanical edits and their descriptions are stored in each cell's `metadata.repo_provenance`.

## Excluded experimental paths

The merged workflow excludes cells that train user embeddings, retrieve neighboring users, tune a learned retriever/reranker, perform LoRA training, extract RAKE keywords, draw paper figures, or run one-off error analysis. These are either outside the paper method or are analysis notebooks rather than the generation pipeline.

## Prompt version reconciliation

The saved output in `Untitled10` prints a prompt containing the complete ranked history. The later Drive snapshot of `prompts/prompts.py` still contains that exact prompt block, but commented out, while the active return omits `history`. The repository simply re-enables the existing original block; it does not introduce a newly written prompt.

## PersonalSum folder

The executable commands in `Untitled10` mount `LaMP_checkpoints/CFRAG`. The Drive `PersonalSum` folder is a separate older summarization project, so its unrelated Python files are not mixed into this repository.
