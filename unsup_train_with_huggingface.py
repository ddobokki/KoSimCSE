import logging
import math
from operator import mod
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random
from data.info import UnsupervisedSimCseFeatures, STSDatasetFeatures
from data.utils import unsupervised_prepare_features, sts_prepare_features
from functools import partial

from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel,
)
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    PreTrainedTokenizerBase,
)
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import (
    cached_property,
    torch_required,
    is_torch_available,
    is_torch_tpu_available,
)
from SimCSE.models import RobertaForCL, BertForCL
from SimCSE.arguments import ModelArguments, DataTrainingArguments, OurTrainingArguments


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

    train_datasets = load_dataset(
        train_extension, data_files=train_data_files, cache_dir="./data/"
    )

    train_datasets = load_dataset(
        train_extension, data_files=train_data_files, cache_dir="./data/"
    )
    valid_datasets = load_dataset(
        valid_extension,
        data_files=eval_data_files,
        cache_dir="./data/",
        delimiter="\t",
    )

    valid_column_names = valid_datasets["dev"].column_names

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    unsup_prepare_features_with_param = partial(
        unsupervised_prepare_features, tokenizer=tokenizer, data_args=data_args
    )
    dev_prepare_features_with_param = partial(
        sts_prepare_features, tokenizer=tokenizer, data_args=data_args
    )

    train_datasets = train_datasets["train"].map(
        unsup_prepare_features_with_param,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=UnsupervisedSimCseFeatures.SENTENCE.value,  # 수정할것 ->datasets.col
        load_from_cache_file=not data_args.overwrite_cache,
    )

    dev_datasets = valid_datasets["dev"].map(
        dev_prepare_features_with_param,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=valid_column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    test_datasets = valid_datasets["test"].map(
        dev_prepare_features_with_param,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=valid_column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )


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
    import SimCSE

    main(model_args, data_args, training_args)
