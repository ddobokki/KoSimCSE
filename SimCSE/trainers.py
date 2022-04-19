"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)
from datasets import load_metric


# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import (
    DataCollator,
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import (
    deepspeed_init,
    deepspeed_reinit,
    is_deepspeed_zero3_enabled,
)
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.optimization import Adafactor, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)


_is_torch_generator_available = False
_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from transformers.trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_gather,
        smp_nested_concat,
    )
from transformers import Trainer

if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

logger = logging.get_logger(__name__)


class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.
    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*)
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs


class CLTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:
        # eval_dataset = eval_dataset if not eval_dataset is None else self.eval_dataset
        self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # print("upper")
        sent1emb = []
        sent2emb = []
        scores = []
        model = self._wrap_model(self.model, training=False)
        model.eval()
        for i, data in enumerate(eval_dataloader):
            with torch.no_grad():
                for k in data:
                    label = data["labels"].view(data["labels"].size(0))
                    scores.extend(label.numpy())
                    # logger.info(data["input_ids"].shape)

                    input_ids1 = (
                        data["input_ids"][:, 0, :]
                        .view(-1, data["input_ids"].size(-1))
                        .to(self.args.device)
                    )
                    input_ids2 = (
                        data["input_ids"][:, 1, :]
                        .view(-1, data["input_ids"].size(-1))
                        .to(self.args.device)
                    )

                    attention_mask1 = (
                        data["attention_mask"][:, 0, :]
                        .view(-1, data["attention_mask"].size(-1))
                        .to(self.args.device)
                    )
                    attention_mask2 = (
                        data["attention_mask"][:, 1, :]
                        .view(-1, data["attention_mask"].size(-1))
                        .to(self.args.device)
                    )

                    if data["token_type_ids"] is not None:
                        token_type_ids1 = (
                            data["token_type_ids"][:, 0, :]
                            .view(-1, data["token_type_ids"].size(-1))
                            .to(self.args.device)
                        )
                        token_type_ids2 = (
                            data["token_type_ids"][:, 1, :]
                            .view(-1, data["token_type_ids"].size(-1))
                            .to(self.args.device)
                        )

                    output1 = model(
                        input_ids=input_ids1,
                        attention_mask=attention_mask1,
                        token_type_ids=token_type_ids1,
                        output_hidden_states=True,
                        return_dict=True,
                        sent_emb=True,
                    )
                    output2 = model(
                        input_ids=input_ids2,
                        attention_mask=attention_mask2,
                        token_type_ids=token_type_ids2,
                        output_hidden_states=True,
                        return_dict=True,
                        sent_emb=True,
                    )
                    pooler_output1 = output1.pooler_output.cpu()
                    pooler_output2 = output2.pooler_output.cpu()
                    sent1emb.append(pooler_output1)
                    sent2emb.append(pooler_output2)
        # print(self.args.device)
        # print(type(self.args.device))
        sent1emb = torch.cat(sent1emb, 0).numpy()
        sent2emb = torch.cat(sent2emb, 0).numpy()

        cos_score = 1 - paired_cosine_distances(sent1emb, sent2emb)
        pearsonr = load_metric("pearsonr").compute
        spearmanr = load_metric("spearmanr").compute
        pearson_corr = pearsonr(predictions=cos_score, references=scores)["pearsonr"]
        spearman_corr = spearmanr(predictions=cos_score, references=scores)["spearmanr"]

        metric = {
            "eval_pearson": pearson_corr,
            "eval_spearman": spearman_corr,
        }
        output.metrics.update(metric)
        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    # def evaluation_loop(
    #     self,
    #     dataloader: DataLoader,
    #     description: str,
    #     prediction_loss_only: Optional[bool] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ) -> EvalLoopOutput:
    #     """
    #     Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
    #     Works both with or without labels.
    #     """
    #     args = self.args

    #     prediction_loss_only = (
    #         prediction_loss_only
    #         if prediction_loss_only is not None
    #         else args.prediction_loss_only
    #     )

    #     # if eval is called w/o train init deepspeed here
    #     if args.deepspeed and not self.deepspeed:

    #         # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
    #         # from the checkpoint eventually
    #         deepspeed_engine, _, _ = deepspeed_init(
    #             self, num_training_steps=0, resume_from_checkpoint=None, inference=True
    #         )
    #         self.model = deepspeed_engine.module
    #         self.model_wrapped = deepspeed_engine
    #         self.deepspeed = deepspeed_engine

    #     model = self._wrap_model(self.model, training=False)

    #     # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
    #     # while ``train`` is running, cast it to the right dtype first and then put on device
    #     if not self.is_in_train:
    #         if args.fp16_full_eval:
    #             model = model.to(dtype=torch.float16, device=args.device)
    #         elif args.bf16_full_eval:
    #             model = model.to(dtype=torch.bfloat16, device=args.device)

    #     batch_size = self.args.eval_batch_size

    #     logger.info(f"***** Running {description} *****")
    #     if has_length(dataloader):
    #         logger.info(f"  Num examples = {self.num_examples(dataloader)}")
    #     else:
    #         logger.info("  Num examples: Unknown")
    #     logger.info(f"  Batch size = {batch_size}")

    #     model.eval()

    #     self.callback_handler.eval_dataloader = dataloader
    #     # Do this before wrapping.
    #     eval_dataset = getattr(dataloader, "dataset", None)

    #     if is_torch_tpu_available():
    #         dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(
    #             args.device
    #         )

    #     if args.past_index >= 0:
    #         self._past = None

    #     # Initialize containers
    #     # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
    #     losses_host = None
    #     preds_host = None
    #     labels_host = None
    #     inputs_host = None

    #     # losses/preds/labels on CPU (final containers)
    #     all_losses = None
    #     all_preds = None
    #     all_labels = None
    #     all_inputs = None
    #     # Will be useful when we have an iterable dataset so don't know its length.

    #     observed_num_examples = 0
    #     # Main evaluation loop
    #     for step, inputs in enumerate(dataloader):
    #         # Update the observed num examples
    #         observed_batch_size = find_batch_size(inputs)
    #         if observed_batch_size is not None:
    #             observed_num_examples += observed_batch_size
    #             # For batch samplers, batch_size is not known by the dataloader in advance.
    #             if batch_size is None:
    #                 batch_size = observed_batch_size

    #         # Prediction step
    #         loss, logits, labels = self.prediction_step(
    #             model, inputs, prediction_loss_only, ignore_keys=ignore_keys
    #         )
    #         inputs_decode = (
    #             inputs["input_ids"] if args.include_inputs_for_metrics else None
    #         ).to(self.args.device)

    #         if is_torch_tpu_available():
    #             xm.mark_step()

    #         # Update containers on host
    #         if loss is not None:
    #             losses = self._nested_gather(loss.repeat(batch_size))
    #             losses_host = (
    #                 losses
    #                 if losses_host is None
    #                 else torch.cat((losses_host, losses), dim=0)
    #             )
    #         if labels is not None:
    #             labels = self._pad_across_processes(labels)
    #             labels = self._nested_gather(labels)
    #             labels_host = (
    #                 labels
    #                 if labels_host is None
    #                 else nested_concat(labels_host, labels, padding_index=-100)
    #             )
    #         if inputs_decode is not None:
    #             inputs_decode = self._pad_across_processes(inputs_decode)
    #             inputs_decode = self._nested_gather(inputs_decode)
    #             inputs_host = (
    #                 inputs_decode
    #                 if inputs_host is None
    #                 else nested_concat(inputs_host, inputs_decode, padding_index=-100)
    #             )
    #         if logits is not None:
    #             logits = self._pad_across_processes(logits)
    #             logits = self._nested_gather(logits)
    #             if self.preprocess_logits_for_metrics is not None:
    #                 logits = self.preprocess_logits_for_metrics(logits, labels)
    #             preds_host = (
    #                 logits
    #                 if preds_host is None
    #                 else nested_concat(preds_host, logits, padding_index=-100)
    #             )
    #         self.control = self.callback_handler.on_prediction_step(
    #             args, self.state, self.control
    #         )

    #         # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
    #         if (
    #             args.eval_accumulation_steps is not None
    #             and (step + 1) % args.eval_accumulation_steps == 0
    #         ):
    #             if losses_host is not None:
    #                 losses = nested_numpify(losses_host)
    #                 all_losses = (
    #                     losses
    #                     if all_losses is None
    #                     else np.concatenate((all_losses, losses), axis=0)
    #                 )
    #             if preds_host is not None:
    #                 logits = nested_numpify(preds_host)
    #                 all_preds = (
    #                     logits
    #                     if all_preds is None
    #                     else nested_concat(all_preds, logits, padding_index=-100)
    #                 )
    #             if inputs_host is not None:
    #                 inputs_decode = nested_numpify(inputs_host)
    #                 all_inputs = (
    #                     inputs_decode
    #                     if all_inputs is None
    #                     else nested_concat(
    #                         all_inputs, inputs_decode, padding_index=-100
    #                     )
    #                 )
    #             if labels_host is not None:
    #                 labels = nested_numpify(labels_host)
    #                 all_labels = (
    #                     labels
    #                     if all_labels is None
    #                     else nested_concat(all_labels, labels, padding_index=-100)
    #                 )

    #             # Set back to None to begin a new accumulation
    #             losses_host, preds_host, inputs_host, labels_host = (
    #                 None,
    #                 None,
    #                 None,
    #                 None,
    #             )

    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of the evaluation loop
    #         delattr(self, "_past")

    #     # Gather all remaining tensors and put them back on the CPU
    #     if losses_host is not None:
    #         losses = nested_numpify(losses_host)
    #         all_losses = (
    #             losses
    #             if all_losses is None
    #             else np.concatenate((all_losses, losses), axis=0)
    #         )
    #     if preds_host is not None:
    #         logits = nested_numpify(preds_host)
    #         all_preds = (
    #             logits
    #             if all_preds is None
    #             else nested_concat(all_preds, logits, padding_index=-100)
    #         )
    #     if inputs_host is not None:
    #         inputs_decode = nested_numpify(inputs_host)
    #         all_inputs = (
    #             inputs_decode
    #             if all_inputs is None
    #             else nested_concat(all_inputs, inputs_decode, padding_index=-100)
    #         )
    #     if labels_host is not None:
    #         labels = nested_numpify(labels_host)
    #         all_labels = (
    #             labels
    #             if all_labels is None
    #             else nested_concat(all_labels, labels, padding_index=-100)
    #         )

    #     # Number of samples
    #     if has_length(eval_dataset):
    #         num_samples = len(eval_dataset)
    #     # The instance check is weird and does not actually check for the type, but whether the dataset has the right
    #     # methods. Therefore we need to make sure it also has the attribute.
    #     elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(
    #         eval_dataset, "num_examples"
    #     ):
    #         num_samples = eval_dataset.num_examples
    #     else:
    #         if has_length(dataloader):
    #             num_samples = self.num_examples(dataloader)
    #         else:  # both len(dataloader.dataset) and len(dataloader) fail
    #             num_samples = observed_num_examples

    #     # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
    #     # samplers has been rounded to a multiple of batch_size, so we truncate.
    #     if all_losses is not None:
    #         all_losses = all_losses[:num_samples]
    #     if all_preds is not None:
    #         all_preds = nested_truncate(all_preds, num_samples)
    #     if all_labels is not None:
    #         all_labels = nested_truncate(all_labels, num_samples)
    #     if all_inputs is not None:
    #         all_inputs = nested_truncate(all_inputs, num_samples)

    #     # Metrics!
    #     if (
    #         self.compute_metrics is not None
    #         and all_preds is not None
    #         and all_labels is not None
    #     ):
    #         if args.include_inputs_for_metrics:
    #             metrics = self.compute_metrics(
    #                 EvalPrediction(
    #                     predictions=all_preds, label_ids=all_labels, inputs=all_inputs
    #                 )
    #             )
    #         else:
    #             metrics = self.compute_metrics(
    #                 EvalPrediction(predictions=all_preds, label_ids=all_labels)
    #             )
    #     else:
    #         metrics = {}
    #     # To be JSON-serializable, we need to remove numpy types or zero-d tensors
    #     metrics = denumpify_detensorize(metrics)

    #     if all_losses is not None:
    #         metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

    #     # Prefix all keys with metric_key_prefix + '_'
    #     for key in list(metrics.keys()):
    #         if not key.startswith(f"{metric_key_prefix}_"):
    #             metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

    #     return EvalLoopOutput(
    #         predictions=all_preds,
    #         label_ids=all_labels,
    #         metrics=metrics,
    #         num_samples=num_samples,
    #     )
