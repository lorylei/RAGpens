import json

from prompts.prompts import get_prompt


class Seq2SeqDataset:
    def __init__(self, input_path, llm_tokenizer, max_length, begin_idx, end_idx):
        with open(input_path, "r", encoding="utf-8") as file:
            self.data = json.load(file)[begin_idx:end_idx]
        self.llm_tokenizer = llm_tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        item = self.data[index]
        return {
            "user_id": item["user_id"],
            "input": get_prompt(
                item["input"], item["retrieval"], self.max_length, self.llm_tokenizer
            ),
            "output": item["output"],
            "original_output": item.get("original_output", ""),
            "rewritten_output": item.get("rewritten_output", ""),
        }

    def __len__(self):
        return len(self.data)
