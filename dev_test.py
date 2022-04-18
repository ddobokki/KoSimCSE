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
from SimCSE.data_collator import SimCseDataCollatorWithPadding
from SimCSE.arguments import ModelArguments, DataTrainingArguments, OurTrainingArguments


logger = logging.getLogger(__name__)


def main(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: OurTrainingArguments,
):
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    print(SimCseDataCollatorWithPadding(tokenizer, model_args, data_args))


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
