import inspect
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .activation_checkpoint import apply_activation_checkpointing
from .data import determine_batch_size, should_perform_precomputation
from .diffusion import (
    default_flow_shift,
    get_scheduler_alphas,
    get_scheduler_sigmas,
    prepare_loss_weights,
    prepare_sigmas,
    prepare_target,
    resolution_dependent_timestep_flow_shift,
)
from .file import delete_files, find_files, string_to_filename
from .hub import save_model_card
from .memory import bytes_to_gigabytes, free_memory, get_memory_statistics, make_contiguous
from .model import resolve_component_cls
from .state_checkpoint import PTDCheckpointManager
from .torch import (
    align_device_and_dtype,
    clip_grad_norm_,
    enable_determinism,
    expand_tensor_dims,
    get_device_info,
    set_requires_grad,
    synchronize_device,
    unwrap_model,
)


def get_parameter_names(obj: Any, method_name: Optional[str] = None) -> Set[str]:
    if method_name is not None:
        obj = getattr(obj, method_name)
    return {name for name, _ in inspect.signature(obj).parameters.items()}


def get_non_null_items(
    x: Union[List[Any], Tuple[Any], Dict[str, Any]]
) -> Union[List[Any], Tuple[Any], Dict[str, Any]]:
    if isinstance(x, dict):
        return {k: v for k, v in x.items() if v is not None}
    if isinstance(x, (list, tuple)):
        return type(x)(v for v in x if v is not None)
