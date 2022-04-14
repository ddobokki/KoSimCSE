from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import (
    LoggingHandler,
    SentenceTransformer,
    InputExample,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import argparse
import pandas as pd
from data.info import (
    STSDatasetFeatures,
    UnsupervisedSimCseFeatures,
    DataName,
    TrainType,
    FileFormat,
)
from typing import List

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout


def sts_tsv_to_input_example(path: str, dataset_name: str) -> List[InputExample]:
    samples = []
    data_path = os.path.join(path, dataset_name)
    data = pd.read_csv(data_path, sep="\t")
    for idx in range(len(data)):
        score = data[STSDatasetFeatures.SCORE.value].iloc[idx] / 5.0
        samples.append(
            InputExample(
                texts=[
                    data[STSDatasetFeatures.SENTENCE1.value].iloc[idx],
                    data[STSDatasetFeatures.SENTENCE2.value].iloc[idx],
                ],
                label=score,
            )
        )
    return samples


def main(args) -> None:
    # train parameters
    model_name = args.model_name
    train_batch_size = args.train_batch_size
    num_epochs = args.num_epochs
    max_seq_length = args.max_seq_length

    # 모델 저장위치
    model_save_path = args.save_path.format(
        model_name, train_batch_size, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # model.start_multi_process_pool()

    train_samples = []
    train_data_path = args.train_data_path
    data_list = os.listdir(train_data_path)
    for data_file in data_list:
        train_path = os.path.join(train_data_path, data_file)
        train_data = pd.read_csv(train_path)
        train_samples.extend(
            train_data[UnsupervisedSimCseFeatures.SENTENCE.value].to_list()
        )
    train_samples = list(map(lambda x: InputExample(texts=[x, x]), train_samples))

    dev_data_name = (
        DataName.PREPROCESS_STS.value + TrainType.DEV.value + FileFormat.TSV.value
    )
    dev_samples = sts_tsv_to_input_example(args.dev_data_path, dev_data_name)

    test_data_name = (
        DataName.PREPROCESS_STS.value + TrainType.TEST.value + FileFormat.TSV.value
    )
    test_samples = sts_tsv_to_input_example(args.test_data_path, test_data_name)

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_samples, batch_size=train_batch_size, name="sts-dev"
    )
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        test_samples, batch_size=train_batch_size, name="sts-test"
    )

    # We train our model using the MultipleNegativesRankingLoss
    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = math.ceil(
        len(train_dataloader) * num_epochs * 0.1
    )  # 10% of train data for warm-up
    evaluation_steps = int(
        len(train_dataloader) * 0.1
    )  # Evaluate every 10% of the data
    logging.info("Training sentences: {}".format(len(train_samples)))
    logging.info("Warmup-steps: {}".format(warmup_steps))
    logging.info("Performance before training")
    dev_evaluator(model)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=num_epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        optimizer_params={"lr": args.learning_rate},
        use_amp=True,  # Set to True, if your GPU supports FP16 cores
    )

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    model = SentenceTransformer(model_save_path)
    test_evaluator(model, output_path=model_save_path)


if __name__ == "__main__":
    # args.add_argument()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="klue/roberta-small", help="model_name"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=256, help="train_batch_size"
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="num_epochs")
    parser.add_argument("--max_seq_length", type=int, default=64, help="max_seq_length")
    parser.add_argument("--save_path", type=str, default="output/unspervised-{}-{}-{}")
    # data path
    parser.add_argument("--train_data_path", type=str, default="data/train")
    parser.add_argument("--dev_data_path", type=str, default="data/dev")
    parser.add_argument("--test_data_path", type=str, default="data/test")
    parser.add_argument("--learning_rate", type=float, default=5e-5)

    args = parser.parse_args()

    main(args=args)
