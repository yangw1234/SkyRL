import torch
import os
from pathlib import Path
from typing import Callable, Optional, Union
from transformers import PretrainedConfig
import safetensors.torch
import numpy as np

def get_param_key(path: tuple, prefix: str = "") -> str:
    "Get the safetensors key for a given model path."
    if path[-1] in {"embedding", "kernel"}:
        path = (*path[:-1], "weight")
    elif path[-1] in {"lora_A", "lora_B"}:
        path = (*path, "weight")
    return prefix + ".".join(map(str, path))

def get_expert_key(path: tuple, expert_idx: int) -> str:
    "Get the safetensors key for an expert weight model path."
    path = tuple(s if s != "experts" else f"experts.{expert_idx}" for s in path)
    return ".".join(map(str, path))


def load_safetensors_pytorch(
    checkpoint_dir: str | os.PathLike,
    config: PretrainedConfig,
    model: torch.nn.Module,
    skip_lora: bool = True,
    prefix: str = "",
    filter_fn: Callable[[tuple], bool] | None = None,
) -> None:
    import torch
    tensors = {}
    for file in Path(checkpoint_dir).glob("*.safetensors"):
        tensors.update(safetensors.torch.load_file(file))
    tensors = {k.removeprefix(prefix): v for k, v in tensors.items()}

    model_params = model.state_dict()
    print(f"Loading model parameters {model}")
    for path, param in model_params.items():
        if filter_fn is not None and not filter_fn(path):
            continue
        # key = get_param_key(path)
        key = path
        path = tuple(path.split('.'))
        # Skip LoRA parameters if requested
        # if skip_lora and ("lora_A" in path or "lora_B" in path or "lora_scaling" in path or "lora_ranks" in path):
        #     continue
        # if "experts" in path:
        #     tensors[key] = np.stack([tensors[get_expert_key(path, i)].T for i in range(config.num_experts)], axis=0)
        # else:
        #     tensors[key] = tensors[key] if "embed_tokens" in path else tensors[key].T
        # if path[-2] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        #     tensors[key] = tensors[key].reshape(param.shape)
        assert param.shape == tensors[key].shape, f"shape mismatch for {key}, expected {param.shape}, got {tensors[key].shape}"
        param.data.copy_(tensors[key])