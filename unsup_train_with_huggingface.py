import logging
from pyexpat import model
from typing import Optional, Union, List, Dict, Tuple
from data.info import UnsupervisedSimCseFeatures, STSDatasetFeatures
from data.utils import unsupervised_prepare_features, sts_prepare_features
from functools import partial

from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import is_main_process
from SimCSE.models import RobertaForCL, BertForCL
from SimCSE.arguments import ModelArguments, DataTrainingArguments, OurTrainingArguments
from SimCSE.data_collator import SimCseDataCollatorWithPadding
from metric import compute_metrics


logger = logging.getLogger(__name__)


def main(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: OurTrainingArguments,
):

    train_data_files = {}
    eval_data_files = {}
    if train_data_files is not None:
        train_data_files["train"] = data_args.train_file
    else:
        # 훈련 데이터가 없으면 종료
        return

    if data_args.dev_file is not None:
        eval_data_files["dev"] = data_args.dev_file

    if data_args.test_file is not None:
        eval_data_files["test"] = data_args.test_file

    train_extension = data_args.train_file.split(".")[-1]  # wiki.csv
    valid_extension = "csv"  # sts.tsv -> 현재 하드코딩 부분

    train_dataset = load_dataset(
        train_extension, data_files=train_data_files, cache_dir="./data/.cache"
    )

    valid_dataset = load_dataset(
        valid_extension,
        data_files=eval_data_files,
        cache_dir="./data/.cache",
        delimiter="\t",
    )

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if "roberta" in model_args.model_name_or_path:
        model = RobertaForCL.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            model_args=model_args,
        )
    elif "bert" in model_args.model_name_or_path:
        model = BertForCL.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            model_args=model_args,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    valid_column_names = valid_dataset["dev"].column_names

    unsup_prepare_features_with_param = partial(
        unsupervised_prepare_features, tokenizer=tokenizer, data_args=data_args
    )
    dev_prepare_features_with_param = partial(
        sts_prepare_features, tokenizer=tokenizer, data_args=data_args
    )

    train_dataset = train_dataset["train"].map(
        unsup_prepare_features_with_param,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=UnsupervisedSimCseFeatures.SENTENCE.value,  # 수정할것 ->datasets.col
        load_from_cache_file=not data_args.overwrite_cache,
    )

    dev_dataset = valid_dataset["dev"].map(
        dev_prepare_features_with_param,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=valid_column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    test_dataset = valid_dataset["test"].map(
        dev_prepare_features_with_param,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=valid_column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    data_collator = (
        default_data_collator
        # if data_args.pad_to_max_length
        # else SimCseDataCollatorWithPadding(
        #     tokenizer=tokenizer, data_args=data_args, model_args=model_args
        # )
    )
    # print(next(iter(dev_dataset)))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    trainer.train()

    if training_args.do_eval:
        eval_result_on_valid_set = trainer.evaluate(dev_dataset)
        logger.info(
            f"Evaluation Result on the valid set! #####\n{eval_result_on_valid_set}"
        )
        eval_result_on_test_set = trainer.evaluate(test_dataset)
        logger.info(
            f"Evaluation Result on the test set! #####\n{eval_result_on_test_set}"
        )
    model.save_pretrained(training_args.output_dir + "/best_model")


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if is_main_process(training_args.local_rank)
        else logging.WARN,
    )

    main(model_args, data_args, training_args)
