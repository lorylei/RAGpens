import jieba
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge


class HeadlineEvaluation:
    def compute_metrics(self, predictions, labels, average=True):
        return compute_bleu_rouge(predictions, labels, average)


def compute_bleu_rouge(predictions, labels, average=True):
    scores = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
    for prediction, label in zip(predictions, labels):
        prediction = prediction.lower()
        label = label.lower()
        hypothesis = list(jieba.cut(prediction))
        reference = list(jieba.cut(label))
        if not hypothesis or not reference:
            rouge_scores = {
                "rouge-1": {"f": 0.0},
                "rouge-2": {"f": 0.0},
                "rouge-l": {"f": 0.0},
            }
        else:
            rouge_scores = Rouge().get_scores(
                " ".join(hypothesis), " ".join(reference)
            )[0]
        for name in ("rouge-1", "rouge-2", "rouge-l"):
            scores[name].append(rouge_scores[name]["f"])
        scores["bleu-4"].append(
            sentence_bleu(
                [list(label)],
                list(prediction),
                smoothing_function=SmoothingFunction().method3,
            )
        )
    if average:
        return {name: float(np.mean(values)) for name, values in scores.items()}
    return scores
