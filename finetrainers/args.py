import argparse
import os
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional

import torch

from .config import SUPPORTED_MODEL_CONFIGS, ModelType, TrainingType
from .logging import get_logger
from .parallel import ParallelBackendEnum
from .utils import get_non_null_items


logger = get_logger()


class BaseArgs:
    r"""
    The arguments for the finetrainers training script.

    For helpful information about arguments, run `python train.py --help`.

    TODO(aryan): add `python train.py --recommend_configs --model_name <model_name>` to recommend
    good training configs for a model after extensive testing.
    TODO(aryan): add `python train.py --memory_requirements --model_name <model_name>` to show
    memory requirements per model, per training type with sensible training settings.

    PARALLEL ARGUMENTS
    ------------------
    parallel_backend (`str`, defaults to `accelerate`):
        The parallel backend to use for training. Choose between ['accelerate', 'ptd'].
    pp_degree (`int`, defaults to `1`):
        The degree of pipeline parallelism.
    dp_degree (`int`, defaults to `1`):
        The degree of data parallelism (number of model replicas).
    dp_shards (`int`, defaults to `-1`):
        The number of data parallel shards (number of model partitions).
    cp_degree (`int`, defaults to `1`):
        The degree of context parallelism.

    MODEL ARGUMENTS
    ---------------
    model_name (`str`):
        Name of model to train. To get a list of models, run `python train.py --list_models`.
    pretrained_model_name_or_path (`str`):
        Path to pretrained model or model identifier from https://huggingface.co/models. The model should be
        loadable based on specified `model_name`.
    revision (`str`, defaults to `None`):
        If provided, the model will be loaded from a specific branch of the model repository.
    variant (`str`, defaults to `None`):
        Variant of model weights to use. Some models provide weight variants, such as `fp16`, to reduce disk
        storage requirements.
    cache_dir (`str`, defaults to `None`):
        The directory where the downloaded models and datasets will be stored, or loaded from.
    tokenizer_id (`str`, defaults to `None`):
        Identifier for the tokenizer model. This is useful when using a different tokenizer than the default from `pretrained_model_name_or_path`.
    tokenizer_2_id (`str`, defaults to `None`):
        Identifier for the second tokenizer model. This is useful when using a different tokenizer than the default from `pretrained_model_name_or_path`.
    tokenizer_3_id (`str`, defaults to `None`):
        Identifier for the third tokenizer model. This is useful when using a different tokenizer than the default from `pretrained_model_name_or_path`.
    text_encoder_id (`str`, defaults to `None`):
        Identifier for the text encoder model. This is useful when using a different text encoder than the default from `pretrained_model_name_or_path`.
    text_encoder_2_id (`str`, defaults to `None`):
        Identifier for the second text encoder model. This is useful when using a different text encoder than the default from `pretrained_model_name_or_path`.
    text_encoder_3_id (`str`, defaults to `None`):
        Identifier for the third text encoder model. This is useful when using a different text encoder than the default from `pretrained_model_name_or_path`.
    transformer_id (`str`, defaults to `None`):
        Identifier for the transformer model. This is useful when using a different transformer model than the default from `pretrained_model_name_or_path`.
    vae_id (`str`, defaults to `None`):
        Identifier for the VAE model. This is useful when using a different VAE model than the default from `pretrained_model_name_or_path`.
    text_encoder_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
        Data type for the text encoder when generating text embeddings.
    text_encoder_2_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
        Data type for the text encoder 2 when generating text embeddings.
    text_encoder_3_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
        Data type for the text encoder 3 when generating text embeddings.
    transformer_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
        Data type for the transformer model.
    vae_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
        Data type for the VAE model.
    layerwise_upcasting_modules (`List[str]`, defaults to `[]`):
        Modules that should have fp8 storage weights but higher precision computation. Choose between ['transformer'].
    layerwise_upcasting_storage_dtype (`torch.dtype`, defaults to `float8_e4m3fn`):
        Data type for the layerwise upcasting storage. Choose between ['float8_e4m3fn', 'float8_e5m2'].
    layerwise_upcasting_skip_modules_pattern (`List[str]`, defaults to `["patch_embed", "pos_embed", "x_embedder", "context_embedder", "^proj_in$", "^proj_out$", "norm"]`):
        Modules to skip for layerwise upcasting. Layers such as normalization and modulation, when casted to fp8 precision
        naively (as done in layerwise upcasting), can lead to poorer training and inference quality. We skip these layers
        by default, and recommend adding more layers to the default list based on the model architecture.

    DATASET ARGUMENTS
    -----------------
    dataset_config (`str`):
        File to a dataset file containing information about training data. This file can contain information about one or
        more datasets in JSON format. The file must have a key called "datasets", which is a list of dictionaries. Each
        dictionary must contain the following keys:
            - "data_root": (`str`)
                The root directory containing the dataset. This parameter must be provided if `dataset_file` is not provided.
            - "dataset_file": (`str`)
                Path to a CSV/JSON/JSONL/PARQUET/ARROW/HF_HUB_DATASET file containing metadata for training. This parameter
                must be provided if `data_root` is not provided.
            - "dataset_type": (`str`)
                Type of dataset. Choose between ['image', 'video'].
            - "id_token": (`str`)
                Identifier token appended to the start of each prompt if provided. This is useful for LoRA-type training
                for single subject/concept/style training, but is not necessary.
            - "image_resolution_buckets": (`List[Tuple[int, int]]`)
                Resolution buckets for image. This should be a list of tuples containing 2 values, where each tuple
                represents the resolution (height, width). All images will be resized to the nearest bucket resolution.
                This parameter must be provided if `dataset_type` is 'image'.
            - "video_resolution_buckets": (`List[Tuple[int, int, int]]`)
                Resolution buckets for video. This should be a list of tuples containing 3 values, where each tuple
                represents the resolution (num_frames, height, width). All videos will be resized to the nearest bucket
                resolution. This parameter must be provided if `dataset_type` is 'video'.
            - "reshape_mode": (`str`)
                All input images/videos are reshaped using this mode. Choose between the following:
                ["center_crop", "random_crop", "bicubic"].
            - "remove_common_llm_caption_prefixes": (`boolean`)
                Whether or not to remove common LLM caption prefixes. See `~constants.py` for the list of common prefixes.
    dataset_shuffle_buffer_size (`int`, defaults to `1`):
        The buffer size for shuffling the dataset. This is useful for shuffling the dataset before training. The default
        value of `1` means that the dataset will not be shuffled.
    precomputation_items (`int`, defaults to `512`):
        Number of data samples to precompute at once for memory-efficient training. The higher this value,
        the more disk memory will be used to save the precomputed samples (conditions and latents).
    precomputation_dir (`str`, defaults to `None`):
        The directory where the precomputed samples will be stored. If not provided, the precomputed samples
        will be stored in a temporary directory of the output directory.
    precomputation_once (`bool`, defaults to `False`):
        Precompute embeddings from all datasets at once before training. This is useful to save time during training
        with smaller datasets. If set to `False`, will save disk space by precomputing embeddings on-the-fly during
        training when required. Make sure to set `precomputation_items` to a reasonable value in line with the size
        of your dataset(s).

    DATALOADER_ARGUMENTS
    --------------------
    See https://pytorch.org/docs/stable/data.html for more information.

    dataloader_num_workers (`int`, defaults to `0`):
        Number of subprocesses to use for data loading. `0` means that the data will be loaded in a blocking manner
        on the main process.
    pin_memory (`bool`, defaults to `False`):
        Whether or not to use the pinned memory setting in PyTorch dataloader. This is useful for faster data loading.

    DIFFUSION ARGUMENTS
    -------------------
    flow_resolution_shifting (`bool`, defaults to `False`):
        Resolution-dependent shifting of timestep schedules.
        [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).
        TODO(aryan): We don't support this yet.
    flow_base_seq_len (`int`, defaults to `256`):
        Base number of tokens for images/video when applying resolution-dependent shifting.
    flow_max_seq_len (`int`, defaults to `4096`):
        Maximum number of tokens for images/video when applying resolution-dependent shifting.
    flow_base_shift (`float`, defaults to `0.5`):
        Base shift for timestep schedules when applying resolution-dependent shifting.
    flow_max_shift (`float`, defaults to `1.15`):
        Maximum shift for timestep schedules when applying resolution-dependent shifting.
    flow_shift (`float`, defaults to `1.0`):
        Instead of training with uniform/logit-normal sigmas, shift them as (shift * sigma) / (1 + (shift - 1) * sigma).
        Setting it higher is helpful when trying to train models for high-resolution generation or to produce better
        samples in lower number of inference steps.
    flow_weighting_scheme (`str`, defaults to `none`):
        We default to the "none" weighting scheme for uniform sampling and uniform loss.
        Choose between ['sigma_sqrt', 'logit_normal', 'mode', 'cosmap', 'none'].
    flow_logit_mean (`float`, defaults to `0.0`):
        Mean to use when using the `'logit_normal'` weighting scheme.
    flow_logit_std (`float`, defaults to `1.0`):
        Standard deviation to use when using the `'logit_normal'` weighting scheme.
    flow_mode_scale (`float`, defaults to `1.29`):
        Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.

    TRAINING ARGUMENTS
    ------------------
    training_type (`str`, defaults to `None`):
        Type of training to perform. Choose between ['lora'].
    seed (`int`, defaults to `42`):
        A seed for reproducible training.
    batch_size (`int`, defaults to `1`):
        Per-device batch size.
    train_steps (`int`, defaults to `1000`):
        Total number of training steps to perform.
    max_data_samples (`int`, defaults to `2**64`):
        Maximum number of data samples observed during training training. If lesser than that required by `train_steps`,
        the training will stop early.
    gradient_accumulation_steps (`int`, defaults to `1`):
        Number of gradients steps to accumulate before performing an optimizer step.
    gradient_checkpointing (`bool`, defaults to `False`):
        Whether or not to use gradient/activation checkpointing to save memory at the expense of slower
        backward pass.
    checkpointing_steps (`int`, defaults to `500`):
        Save a checkpoint of the training state every X training steps. These checkpoints can be used both
        as final checkpoints in case they are better than the last checkpoint, and are also suitable for
        resuming training using `resume_from_checkpoint`.
    checkpointing_limit (`int`, defaults to `None`):
        Max number of checkpoints to store.
    resume_from_checkpoint (`str`, defaults to `None`):
        Whether training should be resumed from a previous checkpoint. Use a path saved by `checkpointing_steps`,
        or `"latest"` to automatically select the last available checkpoint.

    OPTIMIZER ARGUMENTS
    -------------------
    optimizer (`str`, defaults to `adamw`):
        The optimizer type to use. Choose between the following:
            - Torch optimizers: ["adam", "adamw"]
            - Bitsandbytes optimizers: ["adam-bnb", "adamw-bnb", "adam-bnb-8bit", "adamw-bnb-8bit"]
    lr (`float`, defaults to `1e-4`):
        Initial learning rate (after the potential warmup period) to use.
    lr_scheduler (`str`, defaults to `cosine_with_restarts`):
        The scheduler type to use. Choose between ['linear', 'cosine', 'cosine_with_restarts', 'polynomial',
        'constant', 'constant_with_warmup'].
    lr_warmup_steps (`int`, defaults to `500`):
        Number of steps for the warmup in the lr scheduler.
    lr_num_cycles (`int`, defaults to `1`):
        Number of hard resets of the lr in cosine_with_restarts scheduler.
    lr_power (`float`, defaults to `1.0`):
        Power factor of the polynomial scheduler.
    beta1 (`float`, defaults to `0.9`):
    beta2 (`float`, defaults to `0.95`):
    beta3 (`float`, defaults to `0.999`):
    weight_decay (`float`, defaults to `0.0001`):
        Penalty for large weights in the model.
    epsilon (`float`, defaults to `1e-8`):
        Small value to avoid division by zero in the optimizer.
    max_grad_norm (`float`, defaults to `1.0`):
        Maximum gradient norm to clip the gradients.

    VALIDATION ARGUMENTS
    --------------------
    validation_dataset_file (`str`, defaults to `None`):
        Path to a CSV/JSON/PARQUET/ARROW file containing information for validation. The file must contain atleast the
        "caption" column. Other columns such as "image_path" and "video_path" can be provided too. If provided, "image_path"
        will be used to load a PIL.Image.Image and set the "image" key in the sample dictionary. Similarly, "video_path"
        will be used to load a List[PIL.Image.Image] and set the "video" key in the sample dictionary.
        The validation dataset file may contain other attributes specific to inference/validation such as:
            - "height" and "width" and "num_frames": Resolution
            - "num_inference_steps": Number of inference steps
            - "guidance_scale": Classifier-free Guidance Scale
            - ... (any number of additional attributes can be provided. The ModelSpecification::validate method will be
              invoked with the sample dictionary to validate the sample.)
    validation_steps (`int`, defaults to `500`):
        Number of training steps after which a validation step is performed.
    enable_model_cpu_offload (`bool`, defaults to `False`):
        Whether or not to offload different modeling components to CPU during validation.

    MISCELLANEOUS ARGUMENTS
    -----------------------
    tracker_name (`str`, defaults to `finetrainers`):
        Name of the tracker/project to use for logging training metrics.
    push_to_hub (`bool`, defaults to `False`):
        Whether or not to push the model to the Hugging Face Hub.
    hub_token (`str`, defaults to `None`):
        The API token to use for pushing the model to the Hugging Face Hub.
    hub_model_id (`str`, defaults to `None`):
        The model identifier to use for pushing the model to the Hugging Face Hub.
    output_dir (`str`, defaults to `None`):
        The directory where the model checkpoints and logs will be stored.
    logging_dir (`str`, defaults to `logs`):
        The directory where the logs will be stored.
    logging_steps (`int`, defaults to `1`):
        Training logs will be tracked every `logging_steps` steps.
    allow_tf32 (`bool`, defaults to `False`):
        Whether or not to allow the use of TF32 matmul on compatible hardware.
    nccl_timeout (`int`, defaults to `1800`):
        Timeout for the NCCL communication.
    report_to (`str`, defaults to `wandb`):
        The name of the logger to use for logging training metrics. Choose between ['wandb'].
    verbose (`int`, defaults to `1`):
        Whether or not to print verbose logs.
            - 0: Diffusers/Transformers warning logging on local main process only
            - 1: Diffusers/Transformers info logging on local main process only
            - 2: Diffusers/Transformers debug logging on local main process only
            - 3: Diffusers/Transformers debug logging on all processes
    """

    # Parallel arguments
    parallel_backend = ParallelBackendEnum.ACCELERATE
    pp_degree: int = 1
    dp_degree: int = 1
    dp_shards: int = 1
    cp_degree: int = 1
    tp_degree: int = 1

    # Model arguments
    model_name: str = None
    pretrained_model_name_or_path: str = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    cache_dir: Optional[str] = None
    tokenizer_id: Optional[str] = None
    tokenizer_2_id: Optional[str] = None
    tokenizer_3_id: Optional[str] = None
    text_encoder_id: Optional[str] = None
    text_encoder_2_id: Optional[str] = None
    text_encoder_3_id: Optional[str] = None
    transformer_id: Optional[str] = None
    vae_id: Optional[str] = None
    text_encoder_dtype: torch.dtype = torch.bfloat16
    text_encoder_2_dtype: torch.dtype = torch.bfloat16
    text_encoder_3_dtype: torch.dtype = torch.bfloat16
    transformer_dtype: torch.dtype = torch.bfloat16
    vae_dtype: torch.dtype = torch.bfloat16
    layerwise_upcasting_modules: List[str] = []
    layerwise_upcasting_storage_dtype: torch.dtype = torch.float8_e4m3fn
    layerwise_upcasting_skip_modules_pattern: List[str] = [
        "patch_embed",
        "pos_embed",
        "x_embedder",
        "context_embedder",
        "time_embed",
        "^proj_in$",
        "^proj_out$",
        "norm",
    ]

    # Dataset arguments
    dataset_config: str = None
    dataset_shuffle_buffer_size: int = 1
    precomputation_items: int = 512
    precomputation_dir: Optional[str] = None
    precomputation_once: bool = False

    # Dataloader arguments
    dataloader_num_workers: int = 0
    pin_memory: bool = False

    # Diffusion arguments
    flow_resolution_shifting: bool = False
    flow_base_seq_len: int = 256
    flow_max_seq_len: int = 4096
    flow_base_shift: float = 0.5
    flow_max_shift: float = 1.15
    flow_shift: float = 1.0
    flow_weighting_scheme: str = "none"
    flow_logit_mean: float = 0.0
    flow_logit_std: float = 1.0
    flow_mode_scale: float = 1.29

    # Training arguments
    training_type: str = None
    seed: int = 42
    batch_size: int = 1
    train_steps: int = 1000
    max_data_samples: int = 2**64
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    checkpointing_steps: int = 500
    checkpointing_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    enable_slicing: bool = False
    enable_tiling: bool = False

    # Optimizer arguments
    optimizer: str = "adamw"
    lr: float = 1e-4
    lr_scheduler: str = "cosine_with_restarts"
    lr_warmup_steps: int = 0
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    beta3: float = 0.999
    weight_decay: float = 0.0001
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Validation arguments
    validation_dataset_file: Optional[str] = None
    validation_steps: int = 500
    enable_model_cpu_offload: bool = False

    # Miscellaneous arguments
    tracker_name: str = "finetrainers"
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None
    output_dir: str = None
    logging_dir: Optional[str] = "logs"
    logging_steps: int = 1
    allow_tf32: bool = False
    init_timeout: int = 300  # 5 minutes
    nccl_timeout: int = 600  # 10 minutes, considering that validation may be performed
    report_to: str = "wandb"
    verbose: int = 1

    def to_dict(self) -> Dict[str, Any]:
        parallel_arguments = {
            "pp_degree": self.pp_degree,
            "dp_degree": self.dp_degree,
            "dp_shards": self.dp_shards,
            "cp_degree": self.cp_degree,
            "tp_degree": self.tp_degree,
        }

        model_arguments = {
            "model_name": self.model_name,
            "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
            "revision": self.revision,
            "variant": self.variant,
            "cache_dir": self.cache_dir,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_2_id": self.tokenizer_2_id,
            "tokenizer_3_id": self.tokenizer_3_id,
            "text_encoder_id": self.text_encoder_id,
            "text_encoder_2_id": self.text_encoder_2_id,
            "text_encoder_3_id": self.text_encoder_3_id,
            "transformer_id": self.transformer_id,
            "vae_id": self.vae_id,
            "text_encoder_dtype": self.text_encoder_dtype,
            "text_encoder_2_dtype": self.text_encoder_2_dtype,
            "text_encoder_3_dtype": self.text_encoder_3_dtype,
            "transformer_dtype": self.transformer_dtype,
            "vae_dtype": self.vae_dtype,
            "layerwise_upcasting_modules": self.layerwise_upcasting_modules,
            "layerwise_upcasting_storage_dtype": self.layerwise_upcasting_storage_dtype,
            "layerwise_upcasting_skip_modules_pattern": self.layerwise_upcasting_skip_modules_pattern,
        }
        model_arguments = get_non_null_items(model_arguments)

        dataset_arguments = {
            "dataset_config": self.dataset_config,
            "dataset_shuffle_buffer_size": self.dataset_shuffle_buffer_size,
            "precomputation_items": self.precomputation_items,
            "precomputation_dir": self.precomputation_dir,
            "precomputation_once": self.precomputation_once,
        }
        dataset_arguments = get_non_null_items(dataset_arguments)

        dataloader_arguments = {
            "dataloader_num_workers": self.dataloader_num_workers,
            "pin_memory": self.pin_memory,
        }

        diffusion_arguments = {
            "flow_resolution_shifting": self.flow_resolution_shifting,
            "flow_base_seq_len": self.flow_base_seq_len,
            "flow_max_seq_len": self.flow_max_seq_len,
            "flow_base_shift": self.flow_base_shift,
            "flow_max_shift": self.flow_max_shift,
            "flow_shift": self.flow_shift,
            "flow_weighting_scheme": self.flow_weighting_scheme,
            "flow_logit_mean": self.flow_logit_mean,
            "flow_logit_std": self.flow_logit_std,
            "flow_mode_scale": self.flow_mode_scale,
        }

        training_arguments = {
            "training_type": self.training_type,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "train_steps": self.train_steps,
            "max_data_samples": self.max_data_samples,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "checkpointing_steps": self.checkpointing_steps,
            "checkpointing_limit": self.checkpointing_limit,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "enable_slicing": self.enable_slicing,
            "enable_tiling": self.enable_tiling,
        }
        training_arguments = get_non_null_items(training_arguments)

        optimizer_arguments = {
            "optimizer": self.optimizer,
            "lr": self.lr,
            "lr_scheduler": self.lr_scheduler,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_num_cycles": self.lr_num_cycles,
            "lr_power": self.lr_power,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "beta3": self.beta3,
            "weight_decay": self.weight_decay,
            "epsilon": self.epsilon,
            "max_grad_norm": self.max_grad_norm,
        }
        optimizer_arguments = get_non_null_items(optimizer_arguments)

        validation_arguments = {
            "validation_dataset_file": self.validation_dataset_file,
            "validation_steps": self.validation_steps,
            "enable_model_cpu_offload": self.enable_model_cpu_offload,
        }
        validation_arguments = get_non_null_items(validation_arguments)

        miscellaneous_arguments = {
            "tracker_name": self.tracker_name,
            "push_to_hub": self.push_to_hub,
            "hub_token": self.hub_token,
            "hub_model_id": self.hub_model_id,
            "output_dir": self.output_dir,
            "logging_dir": self.logging_dir,
            "logging_steps": self.logging_steps,
            "allow_tf32": self.allow_tf32,
            "init_timeout": self.init_timeout,
            "nccl_timeout": self.nccl_timeout,
            "report_to": self.report_to,
            "verbose": self.verbose,
        }
        miscellaneous_arguments = get_non_null_items(miscellaneous_arguments)

        return {
            "parallel_arguments": parallel_arguments,
            "model_arguments": model_arguments,
            "dataset_arguments": dataset_arguments,
            "dataloader_arguments": dataloader_arguments,
            "diffusion_arguments": diffusion_arguments,
            "training_arguments": training_arguments,
            "optimizer_arguments": optimizer_arguments,
            "validation_arguments": validation_arguments,
            "miscellaneous_arguments": miscellaneous_arguments,
        }

    def extend_args(
        self,
        add_fn: Callable[[argparse.ArgumentParser], None],
        map_fn: Callable[["BaseArgs"], None],
        validate_fn: Callable[["BaseArgs"], None],
    ) -> None:
        if not hasattr(self, "_extended_add_arguments"):
            self._extended_add_arguments = []
        self._extended_add_arguments.append((add_fn, validate_fn, map_fn))

    def parse_args(self):
        _LIST_MODELS = "--list_models"

        parser = argparse.ArgumentParser()

        special_args = [_LIST_MODELS]
        if any(arg in sys.argv for arg in special_args):
            _add_helper_arguments(parser)
            args = parser.parse_args()
            _display_helper_messages(args)
            sys.exit(0)
        else:
            _add_args(parser)
            for extended_add_arg_fns in getattr(self, "_extended_add_arguments", []):
                add_fn, _, _ = extended_add_arg_fns
                add_fn(parser)

            args, remaining_args = parser.parse_known_args()
            logger.debug(f"Remaining unparsed arguments: {remaining_args}")

            mapped_args = _map_to_args_type(args)
            for extended_add_arg_fns in getattr(self, "_extended_add_arguments", []):
                _, _, map_fn = extended_add_arg_fns
                map_fn(args, mapped_args)

            _validate_args(mapped_args)
            for extended_add_arg_fns in getattr(self, "_extended_add_arguments", []):
                _, validate_fn, _ = extended_add_arg_fns
                validate_fn(mapped_args)

            return mapped_args


def _add_args(parser: argparse.ArgumentParser) -> None:
    _add_parallel_arguments(parser)
    _add_model_arguments(parser)
    _add_dataset_arguments(parser)
    _add_dataloader_arguments(parser)
    _add_diffusion_arguments(parser)
    _add_training_arguments(parser)
    _add_optimizer_arguments(parser)
    _add_validation_arguments(parser)
    _add_miscellaneous_arguments(parser)


def _validate_args(args: BaseArgs):
    _validate_model_args(args)
    _validate_dataset_args(args)
    _validate_validation_args(args)


def _add_parallel_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--parallel_backend",
        type=str,
        default=ParallelBackendEnum.ACCELERATE,
        choices=[ParallelBackendEnum.ACCELERATE, ParallelBackendEnum.PTD],
    )
    parser.add_argument("--pp_degree", type=int, default=1)
    parser.add_argument("--dp_degree", type=int, default=1)
    parser.add_argument("--dp_shards", type=int, default=1)
    parser.add_argument("--cp_degree", type=int, default=1)
    parser.add_argument("--tp_degree", type=int, default=1)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model_name", type=str, required=True, choices=[x.value for x in ModelType.__members__.values()]
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--tokenizer_id", type=str, default=None)
    parser.add_argument("--tokenizer_2_id", type=str, default=None)
    parser.add_argument("--tokenizer_3_id", type=str, default=None)
    parser.add_argument("--text_encoder_id", type=str, default=None)
    parser.add_argument("--text_encoder_2_id", type=str, default=None)
    parser.add_argument("--text_encoder_3_id", type=str, default=None)
    parser.add_argument("--transformer_id", type=str, default=None)
    parser.add_argument("--vae_id", type=str, default=None)
    parser.add_argument("--text_encoder_dtype", type=str, default="bf16")
    parser.add_argument("--text_encoder_2_dtype", type=str, default="bf16")
    parser.add_argument("--text_encoder_3_dtype", type=str, default="bf16")
    parser.add_argument("--transformer_dtype", type=str, default="bf16")
    parser.add_argument("--vae_dtype", type=str, default="bf16")
    parser.add_argument("--layerwise_upcasting_modules", type=str, default=[], nargs="+", choices=["transformer"])
    parser.add_argument(
        "--layerwise_upcasting_storage_dtype",
        type=str,
        default="float8_e4m3fn",
        choices=["float8_e4m3fn", "float8_e5m2"],
    )
    parser.add_argument(
        "--layerwise_upcasting_skip_modules_pattern",
        type=str,
        default=["patch_embed", "pos_embed", "x_embedder", "context_embedder", "^proj_in$", "^proj_out$", "norm"],
        nargs="+",
    )


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--dataset_shuffle_buffer_size", type=int, default=1)
    parser.add_argument("--precomputation_items", type=int, default=512)
    parser.add_argument("--precomputation_dir", type=str, default=None)
    parser.add_argument("--precomputation_once", action="store_true")


def _add_dataloader_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")


def _add_diffusion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--flow_resolution_shifting", action="store_true")
    parser.add_argument("--flow_base_seq_len", type=int, default=256)
    parser.add_argument("--flow_max_seq_len", type=int, default=4096)
    parser.add_argument("--flow_base_shift", type=float, default=0.5)
    parser.add_argument("--flow_max_shift", type=float, default=1.15)
    parser.add_argument("--flow_shift", type=float, default=1.0)
    parser.add_argument(
        "--flow_weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
    )
    parser.add_argument("--flow_logit_mean", type=float, default=0.0)
    parser.add_argument("--flow_logit_std", type=float, default=1.0)
    parser.add_argument("--flow_mode_scale", type=float, default=1.29)


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--training_type", type=str, choices=[x.value for x in TrainingType.__members__.values()], required=True
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--max_data_samples", type=int, default=2**64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpointing_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_slicing", action="store_true")
    parser.add_argument("--enable_tiling", action="store_true")


def _add_optimizer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "adam-bnb", "adamw-bnb", "adam-bnb-8bit", "adamw-bnb-8bit"],
    )
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--beta3", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-04)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)


def _add_validation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--validation_dataset_file", type=str, default=None)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--enable_model_cpu_offload", action="store_true")


def _add_miscellaneous_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tracker_name", type=str, default="finetrainers")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="finetrainers-training")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--init_timeout", type=int, default=300)
    parser.add_argument("--nccl_timeout", type=int, default=600)
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2, 3])


def _add_helper_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list_models", action="store_true")


_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


def _map_to_args_type(args: Dict[str, Any]) -> BaseArgs:
    result_args = BaseArgs()

    # Parallel arguments
    result_args.parallel_backend = args.parallel_backend
    result_args.pp_degree = args.pp_degree
    result_args.dp_degree = args.dp_degree
    result_args.dp_shards = args.dp_shards
    result_args.cp_degree = args.cp_degree
    result_args.tp_degree = args.tp_degree

    # Model arguments
    result_args.model_name = args.model_name
    result_args.pretrained_model_name_or_path = args.pretrained_model_name_or_path
    result_args.revision = args.revision
    result_args.variant = args.variant
    result_args.cache_dir = args.cache_dir
    result_args.tokenizer_id = args.tokenizer_id
    result_args.tokenizer_2_id = args.tokenizer_2_id
    result_args.tokenizer_3_id = args.tokenizer_3_id
    result_args.text_encoder_id = args.text_encoder_id
    result_args.text_encoder_2_id = args.text_encoder_2_id
    result_args.text_encoder_3_id = args.text_encoder_3_id
    result_args.transformer_id = args.transformer_id
    result_args.vae_id = args.vae_id
    result_args.text_encoder_dtype = _DTYPE_MAP[args.text_encoder_dtype]
    result_args.text_encoder_2_dtype = _DTYPE_MAP[args.text_encoder_2_dtype]
    result_args.text_encoder_3_dtype = _DTYPE_MAP[args.text_encoder_3_dtype]
    result_args.transformer_dtype = _DTYPE_MAP[args.transformer_dtype]
    result_args.vae_dtype = _DTYPE_MAP[args.vae_dtype]
    result_args.layerwise_upcasting_modules = args.layerwise_upcasting_modules
    result_args.layerwise_upcasting_storage_dtype = _DTYPE_MAP[args.layerwise_upcasting_storage_dtype]
    result_args.layerwise_upcasting_skip_modules_pattern = args.layerwise_upcasting_skip_modules_pattern

    # Dataset arguments
    result_args.dataset_config = args.dataset_config
    result_args.dataset_shuffle_buffer_size = args.dataset_shuffle_buffer_size
    result_args.precomputation_items = args.precomputation_items
    result_args.precomputation_dir = args.precomputation_dir or os.path.join(args.output_dir, "precomputed")
    result_args.precomputation_once = args.precomputation_once

    # Dataloader arguments
    result_args.dataloader_num_workers = args.dataloader_num_workers
    result_args.pin_memory = args.pin_memory

    # Diffusion arguments
    result_args.flow_resolution_shifting = args.flow_resolution_shifting
    result_args.flow_base_seq_len = args.flow_base_seq_len
    result_args.flow_max_seq_len = args.flow_max_seq_len
    result_args.flow_base_shift = args.flow_base_shift
    result_args.flow_max_shift = args.flow_max_shift
    result_args.flow_shift = args.flow_shift
    result_args.flow_weighting_scheme = args.flow_weighting_scheme
    result_args.flow_logit_mean = args.flow_logit_mean
    result_args.flow_logit_std = args.flow_logit_std
    result_args.flow_mode_scale = args.flow_mode_scale

    # Training arguments
    result_args.training_type = args.training_type
    result_args.seed = args.seed
    result_args.batch_size = args.batch_size
    result_args.train_steps = args.train_steps
    result_args.max_data_samples = args.max_data_samples
    result_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    result_args.gradient_checkpointing = args.gradient_checkpointing
    result_args.checkpointing_steps = args.checkpointing_steps
    result_args.checkpointing_limit = args.checkpointing_limit
    result_args.resume_from_checkpoint = args.resume_from_checkpoint
    result_args.enable_slicing = args.enable_slicing
    result_args.enable_tiling = args.enable_tiling

    # Optimizer arguments
    result_args.optimizer = args.optimizer or "adamw"
    result_args.lr = args.lr or 1e-4
    result_args.lr_scheduler = args.lr_scheduler
    result_args.lr_warmup_steps = args.lr_warmup_steps
    result_args.lr_num_cycles = args.lr_num_cycles
    result_args.lr_power = args.lr_power
    result_args.beta1 = args.beta1
    result_args.beta2 = args.beta2
    result_args.beta3 = args.beta3
    result_args.weight_decay = args.weight_decay
    result_args.epsilon = args.epsilon
    result_args.max_grad_norm = args.max_grad_norm

    # Validation arguments
    result_args.validation_dataset_file = args.validation_dataset_file
    result_args.validation_steps = args.validation_steps
    result_args.enable_model_cpu_offload = args.enable_model_cpu_offload

    # Miscellaneous arguments
    result_args.tracker_name = args.tracker_name
    result_args.push_to_hub = args.push_to_hub
    result_args.hub_token = args.hub_token
    result_args.hub_model_id = args.hub_model_id
    result_args.output_dir = args.output_dir
    result_args.logging_dir = args.logging_dir
    result_args.logging_steps = args.logging_steps
    result_args.allow_tf32 = args.allow_tf32
    result_args.init_timeout = args.init_timeout
    result_args.nccl_timeout = args.nccl_timeout
    result_args.report_to = args.report_to
    result_args.verbose = args.verbose

    return result_args


def _validate_model_args(args: BaseArgs):
    if args.training_type == "full-finetune":
        assert (
            "transformer" not in args.layerwise_upcasting_modules
        ), "Layerwise upcasting is not supported for full-finetune training"


def _validate_dataset_args(args: BaseArgs):
    dataset_config = pathlib.Path(args.dataset_config)
    if not dataset_config.exists():
        raise ValueError(f"Dataset config file {args.dataset_config} does not exist.")
    if args.dataset_shuffle_buffer_size < 1:
        raise ValueError("Dataset shuffle buffer size must be greater than 0.")
    if args.precomputation_items < 1:
        raise ValueError("Precomputation items must be greater than 0.")


def _validate_validation_args(args: BaseArgs):
    if args.dp_shards > 1 and args.enable_model_cpu_offload:
        raise ValueError("Model CPU offload is not supported with FSDP at the moment.")


def _display_helper_messages(args: argparse.Namespace):
    if args.list_models:
        print("Supported models:")
        for index, model_name in enumerate(SUPPORTED_MODEL_CONFIGS.keys()):
            print(f"  {index + 1}. {model_name}")
