import torch
import os
from pathlib import Path
from typing import Callable, Optional, Union
from transformers import PretrainedConfig
from tx.tinker.types import LoraConfig
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
    params_dict: torch.nn.ParameterDict,
    prefix: str = "",
    filter_fn: Callable[[tuple], bool] | None = None,
    skip_lora: bool = False,
) -> None:
    import torch
    tensors = {}
    for file in Path(checkpoint_dir).glob("*.safetensors"):
        tensors.update(safetensors.torch.load_file(file))
    tensors = {k.removeprefix(prefix): v for k, v in tensors.items()}

    model_params = params_dict
    # print(f"Loading model parameters {params_dict}")
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
        if key in tensors:
            assert param.shape == tensors[key].shape, f"shape mismatch for {key}, expected {param.shape}, got {tensors[key].shape}"
        else:
            if skip_lora and ("lora_A" in path or "lora_B" in path or "lora_scaling" in path or "lora_ranks" in path):
                continue
            elif "lora_A" in path or "lora_B" in path:
                key = "base_model.model."  + key + ".weight"
                if key not in tensors:
                    continue
                assert param.shape == tensors[key].shape, f"shape mismatch for {key}, expected {param.shape}, got {tensors[key].shape}"
                lora_tensor = tensors[key]
                # if param.shape != lora_tensor.shape:
                #     lora_tensor = lora_tensor.transpose(0, 1)
                #     assert param.shape == lora_tensor.shape, f"shape mismatch for {key} after transpose, expected {param.shape}, got {lora_tensor.shape}"
                tensors[key] = lora_tensor
            else:
                raise KeyError(f"Key {key} not found in checkpoint tensors")
        param.data.copy_(tensors[key])


def extract_adapter_state(
        adapter_index: int,
        all_lora_params: dict[str, torch.Tensor],
        rank: int,
) -> dict[str, torch.Tensor]:
    
    lora_params = dict()

    for name, param in all_lora_params.items():
        if "lora_A" in name:
            lora_params[name] = param[adapter_index, ..., :rank, :]
        elif "lora_B" in name:
            lora_params[name] = param[adapter_index, ..., :rank]
    return lora_params


def load_lora_checkpoint(
    module: torch.nn.Module,
    adapter_config: LoraConfig,
    adapter_index: int,
    checkpoint_path: str
) -> None:
    
    params = module.state_dict()

    adapter_lora_params = extract_adapter_state(adapter_index,
                                                params,
                                                adapter_config.rank)
    
    assert len(adapter_lora_params) > 0, "No LoRA parameters found to load"

    load_safetensors_pytorch(checkpoint_path,
                             adapter_lora_params)
    
    for name, param in adapter_lora_params.items():
        if "lora_scaling" in name:
            param[adapter_index] = adapter_config.alpha / adapter_config.rank

        if "lora_ranks" in name:
            param[adapter_index] = adapter_config.rank

    # all lora_B in params should have norms > 0
    # for name, param in adapter_lora_params.items():
    #     if "lora_B" in name:
    #         norm = param.norm().item()
    #         if norm == 0.0:
    #             raise ValueError(f"LoRA parameter {name} has zero norm after loading from {checkpoint_path}")
    #         else:
    #             print(f"LoRA parameter {name} loaded with norm {norm:.6f}")