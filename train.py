import logging

from datasets import load_from_disk

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
)
from transformers.trainer_utils import is_main_process
from SimCSE.models import RobertaForCL, BertForCL
from SimCSE.arguments import ModelArguments, DataTrainingArguments, OurTrainingArguments
from SimCSE.data_collator import SimCseDataCollatorWithPadding
from SimCSE.trainers import CLTrainer


logger = logging.getLogger(__name__)


def main(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: OurTrainingArguments,
):

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

    train_dataset = load_from_disk(data_args.train_file)
    valid_datset = load_from_disk(data_args.dev_file)
    dev_dataset = valid_datset["dev"]
    test_dataset = valid_datset["test"]

    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else SimCseDataCollatorWithPadding(
            tokenizer=tokenizer, data_args=data_args, model_args=model_args
        )
    )
    # print(next(iter(dev_dataset)))
    # compute_metric_with_model = partial(compute_metrics, args=training_args)

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
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
