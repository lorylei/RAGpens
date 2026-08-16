import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from prompts.pre_process import get_query


class Retriever:
    def __init__(self, opts):
        self.retriever = opts.retriever
        self.topk = opts.topk
        self.output_path = Path(opts.output_path)
        self.device = opts.device
        self.batch_size = opts.batch_size
        self.title_weight = opts.title_weight
        self.use_title_only = opts.use_title_only
        self.rng = np.random.default_rng(opts.seed)
        with open(opts.input_path, "r", encoding="utf-8") as file:
            self.dataset = json.load(file)[opts.begin_idx:opts.end_idx]
        self.dense_model = None
        if self.retriever == "dense":
            from sentence_transformers import SentenceTransformer

            self.dense_model = SentenceTransformer(opts.dense_model, device=self.device)

    def build_document(self, item):
        title = (item.get("title") or "").strip()
        body = (item.get("text") or "").strip()
        if self.use_title_only:
            return (title + " ") * max(1, self.title_weight)
        if title and body:
            return (((title + " ") * max(1, self.title_weight)) + "\n\n" + body).strip()
        return title or body

    def retrieve(self, sample):
        profile = sample.get("profile", [])
        if not profile:
            return []
        topk = min(self.topk, len(profile))
        query = get_query(sample.get("input", ""))

        if self.retriever == "bm25":
            from rank_bm25 import BM25Okapi

            model = BM25Okapi([self.build_document(item).split() for item in profile])
            scores = model.get_scores(query.split())
            indices = np.argsort(scores)[::-1][:topk]
        elif self.retriever == "dense":
            documents = [self.build_document(item) for item in profile]
            query_embedding = self.dense_model.encode(
                f"query: {query}", normalize_embeddings=True, device=self.device
            )
            document_embeddings = self.dense_model.encode(
                [f"passage: {text}" for text in documents],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                device=self.device,
            )
            scores = np.dot(document_embeddings, query_embedding)
            indices = np.argsort(scores)[::-1][:topk]
        else:
            indices = self.rng.choice(len(profile), size=topk, replace=False)
            scores = None

        retrieval = []
        for rank, index in enumerate(indices, start=1):
            item = copy.deepcopy(profile[int(index)])
            item["rank"] = rank
            if scores is not None:
                item["score"] = float(scores[int(index)])
            retrieval.append(item)
        return retrieval

    def run(self):
        results = []
        for sample in tqdm(self.dataset, desc=f"{self.retriever} retrieval"):
            profile = sample.get("profile", [])
            topics = [item["topic"] for item in profile if item.get("topic")]
            result = {
                "id": sample.get("id", ""),
                "input": sample.get("input", ""),
                "retrieval": self.retrieve(sample),
                "output": sample.get("output", ""),
                "user_id": sample.get("user_id", ""),
                "topic": sample.get("topic", ""),
                "topic_top5": [topic for topic, _ in Counter(topics).most_common(5)],
            }
            for key in ("original_output", "rewritten_output"):
                if key in sample:
                    result[key] = sample[key]
            results.append(result)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved {len(results)} samples to {self.output_path}")
