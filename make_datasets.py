from data.info import UnsupervisedSimCseFeatures
from functools import partial

from datasets import load_dataset

from transformers import HfArgumentParser
from transformers import AutoTokenizer
from transformers.trainer_utils import is_main_process
from SimCSE.arguments import DataTrainingArguments, ModelArguments
from data.prepare_func import unsupervised_prepare_features, sts_prepare_features


def main(model_args: ModelArguments, data_args: DataTrainingArguments):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

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
        train_extension,
        data_files=train_data_files,
        cache_dir=data_args.save_dir + "/.cache",
    )

    valid_dataset = load_dataset(
        valid_extension,
        data_files=eval_data_files,
        cache_dir=data_args.save_dir + "/.cache",
        delimiter="\t",
    )

    unsup_prepare_features_with_param = partial(
        unsupervised_prepare_features, tokenizer=tokenizer, data_args=data_args
    )
    dev_prepare_features_with_param = partial(
        sts_prepare_features, tokenizer=tokenizer, data_args=data_args
    )
    valid_column_names = valid_dataset["dev"].column_names

    train_dataset = (
        train_dataset["train"]
        .map(
            unsup_prepare_features_with_param,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=UnsupervisedSimCseFeatures.SENTENCE.value,  # 수정할것 ->datasets.col
            load_from_cache_file=not data_args.overwrite_cache,
        )
        .save_to_disk(data_args.save_dir + "/train")
    )

    valid_dataset["dev"] = valid_dataset["dev"].map(
        dev_prepare_features_with_param,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=valid_column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    valid_dataset["test"] = valid_dataset["test"].map(
        dev_prepare_features_with_param,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=valid_column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    valid_dataset.save_to_disk(data_args.save_dir + "/validation")


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    main(model_args, data_args)
