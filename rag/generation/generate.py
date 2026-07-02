import sys

sys.path.append('.')
import argparse
import json
import os

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from data.datasets import Seq2SeqDataset
from metrics.eval_metrics import LaMPEvaluation
from prompts.post_process import load_post_process_function

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


def get_text_template(tokenizer: AutoTokenizer, prompt):
    message = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(message,
                                         tokenize=False,
                                         add_generation_prompt=True)


parser = argparse.ArgumentParser()

parser.add_argument("--CUDA_VISIBLE_DEVICES", default='0,1')
parser.add_argument("--random_seed", type=int, default=2024)

parser.add_argument("--base_addr", default='')
parser.add_argument("--task", default="LaMP_4")
parser.add_argument("--input_path", default='train/bm25/bge-base-en-v1.5_5/')
parser.add_argument("--source",
                    default='bge-reranker-base/20250420-162138')
parser.add_argument("--file_name",
                    default='20241009-122157_user-6_20241009-120906')

parser.add_argument("--model_name",
                    default="Llama-3.1-8B-Instruct",
                    choices=['Meta-Llama-3-8B-Instruct', 'Qwen2.5-7B-Instruct','Llama-3.1-8B-Instruct','granite-3.3-8b-instruct'])
parser.add_argument("--begin_idx", type=int, default=0)
parser.add_argument("--end_idx", type=int, default=206000)

# Generation Config
parser.add_argument("--max_new_tokens", type=int, default=64)
parser.add_argument("--cutoff_len", type=int, default=36000)

if __name__ == "__main__":
    opts = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.CUDA_VISIBLE_DEVICES

    opts.base_addr = os.path.join(f"{opts.model_name}_outputs/", opts.task,
                                  opts.input_path)
    opts.model_path = 'meta-llama/{}'.format(opts.model_name)
    # opts.model_path = 'Qwen/{}'.format(opts.model_name)
    for flag, value in opts.__dict__.items():
        print('{}: {}'.format(flag, value))

    tokenizer = AutoTokenizer.from_pretrained(opts.model_path,use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    opts.input_file = os.path.join(opts.base_addr, opts.source,
                                   f'{opts.file_name}.json')
    out_file_name = f"{opts.file_name}_vllm_new-{opts.max_new_tokens}"
    opts.output_dir = os.path.join(opts.base_addr, opts.source, out_file_name)
    eval_dataset = Seq2SeqDataset(opts.input_file,
                                  task=opts.task,
                                  llm_tokenizer=tokenizer,
                                  max_length=opts.cutoff_len,
                                  begin_idx=opts.begin_idx,
                                  end_idx=opts.end_idx)
    
    ############################################
    # with open(opts.input_file, "r", encoding="utf-8") as f:
    #   raw_data = json.load(f)

    # selected_indices = []
    # for i, record in enumerate(raw_data):
    #     history_str = "".join([
    #         f"'text': '{p.get('text','')}' 'title': '{p.get('title','')}'\n"
    #         for p in record["retrieval"]
    #     ])
    #     prompt_str = eval_dataset[i]["input"]

    #     profile_tokens = len(tokenizer.encode(history_str))
    #     total_tokens = len(tokenizer.encode(prompt_str))
    #     ratio = profile_tokens / max(1, total_tokens)

    #     if 0.8 <= ratio :
    #         selected_indices.append(i)
    # prompts = [
    # get_text_template(tokenizer, eval_dataset[i]['input'])
    # for i in selected_indices
    # ]
    # ground_truth = [eval_dataset[i]['output'] for i in selected_indices]
    ############################################
    
    post_process_fun = load_post_process_function(opts.task)
    generate_results = []
    all_scores = None
    # max_prompt_len = 8100
    # filtered_prompts = []
    # filtered_indices = []
    # for idx, rec in enumerate(eval_dataset):
    #   n_tokens = len(tokenizer.encode(rec["input"]))
    #   if n_tokens <= max_prompt_len:
    #     filtered_prompts.append(rec["input"])
    #     filtered_indices.append(idx)
    # prompts = filtered_prompts
    prompts = [
        get_text_template(tokenizer, eval_dataset[i]['input'])
        for i in range(len(eval_dataset))
    ]
    ###########
    # def truncate_prompt(text, tokenizer, cutoff_len, max_new_tokens):
    #   ids = tokenizer(text, add_special_tokens=False).input_ids   
    #   keep = cutoff_len - max_new_tokens
    #   if keep <= 0:
    #       raise ValueError("cutoff_len 必须大于 max_new_tokens")
    #   if len(ids) > keep:
    #       ids = ids[-keep:]   
    #   return tokenizer.decode(ids, skip_special_tokens=False)
    # prompts = []
    # for i in range(len(eval_dataset)):
    #     raw_prompt = get_text_template(tokenizer, eval_dataset[i]['input'])
    #     truncated = truncate_prompt(raw_prompt, tokenizer, opts.cutoff_len, opts.max_new_tokens)
    #     prompts.append(truncated)
    ###########

    
    # target_ctx = opts.cutoff_len  # 40000
    # orig_ctx = 32768
    # yarn_factor = max(1.0, float(target_ctx) / orig_ctx)  # ~1.22 for 40k
    
    llm = LLM(model=opts.model_path,
              gpu_memory_utilization=0.70,
              max_model_len=opts.cutoff_len,
              max_num_seqs=32
              )

    sampling_params = SamplingParams(seed=opts.random_seed,
                                     temperature=0,
                                     best_of=1,
                                     max_tokens=opts.max_new_tokens)

    model_outputs = llm.generate(prompts, sampling_params)
    model_preds = [x.outputs[0].text for x in model_outputs]
    processed_preds = post_process_fun(model_preds)
    # processed_preds = model_preds
    # ground_truth = [
    #       eval_dataset[i]['output'] for i in filtered_indices
    # ]

    ground_truth = [
        eval_dataset[i]['output'] for i in range(len(eval_dataset))
    ]

    eval_method = LaMPEvaluation(opts.task)
    pred_scores = eval_method.compute_metrics(processed_preds,
                                              ground_truth,
                                              avg=False)

    for idx in tqdm(range(len(prompts))):
        data = eval_dataset[idx]
        save_dict = {
            "user_id": data['user_id'],
            "input": data['input'],
            "output": model_preds[idx],
            "predict": processed_preds[idx],
            "label": data['output']
        }
        scores = {k: v[idx] for k, v in pred_scores.items()}
        if all_scores is None:
            all_scores = {k: [v] for k, v in scores.items()}
        else:
            for k in all_scores.keys():
                all_scores[k].append(scores[k])
        save_dict.update(scores)
        generate_results.append(save…14002 tokens truncated…on")

        self.dataset = json.load(open(file_path, 'r'))
        print("orig datasize:{}".format(len(self.dataset)))
        self.dataset = self.dataset[opts.begin_idx:opts.end_idx]

    def load_user(self, opts):
        opts.user_vocab_path = os.path.join(opts.data_addr,
                                            f"dev/{opts.source}")
        vocab_addr = opts.user_vocab_path

        opts.user_emb_path = os.path.join(opts.data_addr,
                                          f"dev/{opts.source}/user_emb",
                                          opts.user_emb_path)
        self.user_emb_path = opts.user_emb_path
        self.user_emb_name = '.'.join(
            os.path.basename(self.user_emb_path).split('.')[:-1])
        self.user_embedding = torch.load(self.user_emb_path).to(self.device)

        with open(os.path.join(vocab_addr, 'user_vocab.pkl'), 'rb') as file:
            self.user_vocab = pickle.load(file)

        with open(os.path.join(vocab_addr, 'user2id.pkl'), 'rb') as file:
            self.user2id = pickle.load(file)

        assert self.user_embedding.shape[0] == len(self.user_vocab)
        assert len(self.user_vocab) == len(self.user2id)

    def run(self):
        results = []
        for idx, data in enumerate(tqdm(self.dataset)):
            rerank_profs, rerank_scores = self.rerank_topk(
                data['query'], data['retrieval'], data['user_id'])

            new_cur_reranked = []
            for prof_idx, profile in enumerate(rerank_profs):
                cur_profile = copy.deepcopy(profile)
                cur_profile['rerank_score'] = rerank_scores[prof_idx]
                new_cur_reranked.append(cur_profile)

            results.append({
                "input": data['input'],
                "query": data['query'],
                "output": data['output'],
                "user_id": data['user_id'],
                "retrieval": new_cur_reranked
            })

        if self.rerank_type == 'bm25' or self.rerank_type == 'direct':
            output_addr = os.path.join(
                self.output_addr, self.data_split, self.source,
                self.input_source, f"{self.rerank_type}_rerank_{self.topk}")
        elif self.rerank_type == 'cross' or self.rerank_type == 'cross_tune':
            output_addr = os.path.join(self.output_addr, self.data_split,
                                       self.source, self.input_source,
                                       self.result_name)
        elif self.rerank_type == 'optim':
            output_addr = os.path.join(
                self.output_addr, self.data_split, self.source,
                self.input_source,
                f"optim_rerank_{self.topk}_{self.optim_metric}")
        if not os.path.exists(output_addr):
            os.makedirs(output_addr)

        result_path = os.path.join(output_addr, f"{self.input_file}.json")

        print("save file to: {}".format(result_path))
        with open(result_path, 'w') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

    @torch.no_grad()
    def rerank_topk(self, query, profile, user):
        corpus = self.get_corpus(profile, self.use_date)

        if self.rerank_type == 'direct':
            scores = [x['score'] for x in profile]
            top_n = np.argsort(scores)[::-1][:self.topk]
            top_n_scores = [scores[i] for i in top_n]
            selected_profs = [profile[i] for i in top_n]
        elif self.rerank_type == 'bm25':
            bm25 = BM25Okapi([x.split() for x in corpus])
            scores = bm25.get_scores(query.split())
            top_n = np.argsort(scores)[::-1][:self.topk]
            top_n_scores = [scores[i] for i in top_n]
            selected_profs = [profile[i] for i in top_n]
        elif self.rerank_type == 'cross' or self.rerank_type == 'cross_tune':
            selected_profs, top_n_scores = self.reranker.rerank_topk(
                corpus, profile, query, user, self.topk)

        return selected_profs, top_n_scores

