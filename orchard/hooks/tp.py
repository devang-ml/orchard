# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

from torch.distributed import _functional_collectives as funcol

from orchard.networks.transformer import Transformer

def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

def _feedforward_forward_hook(module, input, output):
    world_size = _get_world_size()
    return funcol.all_reduce(output, "sum", list(range(world_size)))

def _attention_forward_hook(module, input, output):
    world_size = _get_world_size()
    return funcol.all_reduce(output[0], "sum", list(range(world_size)))

def TP_register_hooks(model: Transformer) -> None:
    for block in model.layers:
        block.attention.register_forward_hook(_attention_forward_hook)
        block.feed_forward.register_forward_hook(_feedforward_forward_hook)
