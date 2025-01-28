# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Optional, Tuple, Type

import torch
from accelerate.logging import get_logger

from ..constants import FINETRAINERS_LOG_LEVEL
from .hooks import HookRegistry, ModelHook


logger = get_logger("finetrainers")  # pylint: disable=invalid-name
logger.setLevel(FINETRAINERS_LOG_LEVEL)


# fmt: off
_SUPPORTED_PYTORCH_LAYERS = (
    torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
    torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
    torch.nn.Linear,
)

_DEFAULT_SKIP_MODULES_PATTERN = ("pos_embed", "patch_embed", "norm")
# fmt: on


class LayerwiseUpcastingHook(ModelHook):
    r"""
    A hook that casts the weights of a module to a high precision dtype for computation, and to a low precision dtype
    for storage. This process may lead to quality loss in the output, but can significantly reduce the memory
    footprint.
    """

    _is_stateful = False

    def __init__(self, storage_dtype: torch.dtype, compute_dtype: torch.dtype, non_blocking: bool) -> None:
        self.storage_dtype = storage_dtype
        self.compute_dtype = compute_dtype
        self.non_blocking = non_blocking

    def initialize_hook(self, module: torch.nn.Module):
        module.to(dtype=self.storage_dtype, non_blocking=self.non_blocking)
        return module

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        module.to(dtype=self.compute_dtype, non_blocking=self.non_blocking)
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output):
        module.to(dtype=self.storage_dtype, non_blocking=self.non_blocking)
        return output


def apply_layerwise_upcasting(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    skip_modules_pattern: Optional[Tuple[str]] = _DEFAULT_SKIP_MODULES_PATTERN,
    skip_modules_classes: Optional[Tuple[Type[torch.nn.Module]]] = None,
    non_blocking: bool = False,
    _prefix: str = "",
) -> None:
    r"""
    Applies layerwise upcasting to a given module. The module expected here is a Diffusers ModelMixin but it can be any
    nn.Module using diffusers layers or pytorch primitives.
    Args:
        module (`torch.nn.Module`):
            The module whose leaf modules will be cast to a high precision dtype for computation, and to a low
            precision dtype for storage.
        storage_dtype (`torch.dtype`):
            The dtype to cast the module to before/after the forward pass for storage.
        compute_dtype (`torch.dtype`):
            The dtype to cast the module to during the forward pass for computation.
        skip_modules_pattern (`Tuple[str]`, defaults to `["pos_embed", "patch_embed", "norm"]`):
            A list of patterns to match the names of the modules to skip during the layerwise upcasting process.
        skip_modules_classes (`Tuple[Type[torch.nn.Module]]`, defaults to `None`):
            A list of module classes to skip during the layerwise upcasting process.
        non_blocking (`bool`, defaults to `False`):
            If `True`, the weight casting operations are non-blocking.
    """
    if skip_modules_classes is None and skip_modules_pattern is None:
        apply_layerwise_upcasting_hook(module, storage_dtype, compute_dtype, non_blocking)
        return

    should_skip = (skip_modules_classes is not None and isinstance(module, skip_modules_classes)) or (
        skip_modules_pattern is not None and any(re.search(pattern, _prefix) for pattern in skip_modules_pattern)
    )
    if should_skip:
        logger.debug(f'Skipping layerwise upcasting for layer "{_prefix}"')
        return

    if isinstance(module, _SUPPORTED_PYTORCH_LAYERS):
        logger.debug(f'Applying layerwise upcasting to layer "{_prefix}"')
        apply_layerwise_upcasting_hook(module, storage_dtype, compute_dtype, non_blocking)
        return

    for name, submodule in module.named_children():
        layer_name = f"{_prefix}.{name}" if _prefix else name
        apply_layerwise_upcasting(
            submodule,
            storage_dtype,
            compute_dtype,
            skip_modules_pattern,
            skip_modules_classes,
            non_blocking,
            _prefix=layer_name,
        )


def apply_layerwise_upcasting_hook(
    module: torch.nn.Module, storage_dtype: torch.dtype, compute_dtype: torch.dtype, non_blocking: bool
) -> None:
    r"""
    Applies a `LayerwiseUpcastingHook` to a given module.
    Args:
        module (`torch.nn.Module`):
            The module to attach the hook to.
        storage_dtype (`torch.dtype`):
            The dtype to cast the module to before the forward pass.
        compute_dtype (`torch.dtype`):
            The dtype to cast the module to during the forward pass.
        non_blocking (`bool`):
            If `True`, the weight casting operations are non-blocking.
    """
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = LayerwiseUpcastingHook(storage_dtype, compute_dtype, non_blocking)
    registry.register_hook(hook, "layerwise_upcasting")
