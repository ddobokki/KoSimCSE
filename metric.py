from datasets import load_metric
from transformers.trainer_utils import EvalPrediction

pearsonr = load_metric("pearsonr").compute
spearmanr = load_metric("spearmanr").compute


def compute_metrics(pred: EvalPrediction):

    references = pred.label_ids
    predictions = pred.predictions
    print(references.shape)
    print("---")
    print(predictions.shape)

    pearson_corr = pearsonr(predictions=predictions, references=references)[0].item()
    spearman_corr = spearmanr(predictions=predictions, references=references)[0].item()
    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
    }
