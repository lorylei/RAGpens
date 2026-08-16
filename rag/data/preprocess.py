import argparse
import json
from pathlib import Path

import pandas as pd


def split_values(value, separator):
    return [item.strip() for item in str(value or "").split(separator) if item.strip()]


def split_rewrites(value):
    return [item.strip() for item in str(value or "").split(";;")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--news_file", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--target", choices=["original", "rewritten"], required=True)
    parser.add_argument("--max_samples", type=int)
    return parser.parse_args()


def build_samples(news_file, test_file, target, max_samples=None):
    news_df = pd.read_csv(news_file, sep="\t", dtype=str, keep_default_na=False)
    test_df = pd.read_csv(test_file, sep="\t", dtype=str, keep_default_na=False)
    news_columns = {"News ID", "News body", "Headline", "Topic"}
    test_columns = {"userid", "clicknewsID", "posnewID"}
    if target == "rewritten":
        test_columns.add("rewrite_titles")
    if missing := news_columns - set(news_df.columns):
        raise ValueError(f"Missing news columns: {sorted(missing)}")
    if missing := test_columns - set(test_df.columns):
        raise ValueError(f"Missing test columns: {sorted(missing)}")

    news_map = {
        row["News ID"].strip(): {
            "text": row["News body"].strip(),
            "title": row["Headline"].strip(),
            "topic": row["Topic"].strip(),
            "id": row["News ID"].strip(),
        }
        for _, row in news_df.iterrows()
        if row["News ID"].strip()
    }

    samples = []
    mismatched_rows = 0
    missing_profile = 0
    missing_target = 0
    for _, row in test_df.iterrows():
        user_id = row["userid"].strip()
        click_ids = split_values(row["clicknewsID"], ",")
        positive_ids = split_values(row["posnewID"], ",")
        rewrites = split_rewrites(row.get("rewrite_titles", ""))
        if target == "rewritten" and len(positive_ids) != len(rewrites):
            mismatched_rows += 1
        pair_count = len(positive_ids) if target == "original" else min(len(positive_ids), len(rewrites))

        for index, news_id in enumerate(positive_ids[:pair_count]):
            news = news_map.get(news_id)
            if news is None:
                missing_target += 1
                continue
            profile = []
            for click_id in click_ids:
                if click_id == news_id:
                    continue
                clicked = news_map.get(click_id)
                if clicked is None:
                    missing_profile += 1
                    continue
                profile.append({**clicked, "user_id": user_id})
            article = news["text"] or news["title"]
            samples.append({
                "id": news_id,
                "input": f"Generate a headline for the following article: {article}",
                "profile": profile,
                "user_id": user_id,
                "topic": news["topic"],
                "output": news["title"] if target == "original" else rewrites[index],
                "original_output": news["title"],
                "rewritten_output": rewrites[index] if index < len(rewrites) else "",
            })
            if max_samples and len(samples) >= max_samples:
                return samples, mismatched_rows, missing_profile, missing_target
    return samples, mismatched_rows, missing_profile, missing_target


def main():
    args = parse_args()
    samples, mismatches, missing_profile, missing_target = build_samples(
        args.news_file, args.test_file, args.target, args.max_samples
    )
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(samples)} samples to {output_file}")
    print(f"mismatched_rows={mismatches}, missing_profile={missing_profile}, missing_target={missing_target}")


if __name__ == "__main__":
    main()
