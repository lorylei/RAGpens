# Personalizing News Headlines with RAG

This repository contains the PENS pipeline used in *Personalizing News Headlines with Retrieval-Augmented Generation*. It uses pretrained instruction models directly and contains only preprocessing, retrieval, generation, and evaluation.

The root notebook downloads the PENS test files, builds both original-headline and rewritten-headline references, retrieves each user's history, generates headlines, and evaluates them against both references.

## Retrieval

The three retrievers used in the paper are retained:

- `bm25`: BM25 over each clicked article's title and body;
- `dense`: `BAAI/bge-base-en-v1.5` with `query:` and `passage:` prefixes;
- `random`: seeded sampling without replacement.

All methods rank or sample only within the current user's clicked history. The selected top-k records are concatenated into the generation prompt in retrieval order.

## Run

Open `pens_rag_pipeline.ipynb` from the repository root and run its two sections. The paper settings are already shown in the configuration cell: k up to 11, 70k context length, 64 output tokens, temperature 0, best-of 1, 70% GPU-memory utilization, and up to 32 concurrent sequences.

The notebook installs the tested package versions and removes Colab's preinstalled TorchAudio, which otherwise conflicts with vLLM's PyTorch build. TorchAudio is not used by this text pipeline.

The same stages can be called directly:

```bash
python rag/data/preprocess.py --news_file data/news.tsv --test_file data/personalized_test.tsv --output_file data/rank_merge_rewritten.json --target rewritten

python rag/ranking.py --input_path data/rank_merge_rewritten.json --output_path data/rewritten/bm25_8/point_base_bm25.json --retriever bm25 --topk 8

python rag/generation/test.py --input_file data/rewritten/bm25_8/point_base_bm25.json --output_dir outputs/llama31/bm25_8 --model_path meta-llama/Llama-3.1-8B-Instruct
```

`rag/generation/generate.py` is kept as a compatibility entry point for the original Untitled10 command and runs the same code as `test.py`.

## Layout

```text
pens_rag_pipeline.ipynb  complete Colab pipeline
rag/data/preprocess.py   PENS data preprocessing 
rag/ranking.py           BM25, BGE, and random retrieval
rag/generation/test.py   vLLM generation and ROUGE/BLEU evaluation
rag/generation/generate.py  
rag/prompts/             RAG prompt and output post-processing
rag/metrics/             headline evaluation
```

The PENS dataset is distributed for research use under the Microsoft Research License Terms. The notebook uses the `THEATLAS/PENS` Hugging Face mirror for automated test-file download.
