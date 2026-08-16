import argparse

from runners.RetrievalRunner import Retriever


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--retriever", choices=["bm25", "dense", "random"], required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--begin_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=20600)
    parser.add_argument("--dense_model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--title_weight", type=int, default=1)
    parser.add_argument("--use_title_only", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    for flag, value in vars(opts).items():
        print(f"{flag}: {value}")
    Retriever(opts).run()
