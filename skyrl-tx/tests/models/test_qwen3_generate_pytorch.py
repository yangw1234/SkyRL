from transformers import AutoTokenizer, PretrainedConfig, AutoModelForCausalLM
from tx.models.configs import Qwen3Config
from tx.models.pt_models.qwen3 import Qwen3ForCausalLM
from tx.utils.models_torch import load_safetensors_pytorch, load_lora_checkpoint
import torch
from peft import PeftModelForCausalLM
from tx.tinker.types import LoraConfig
def test_qwen3_generate():
    """Test batched text generation with KV caching matches HuggingFace."""
    model_name = "Qwen/Qwen3-0.6B"
    adapter_name = "haizelabs/j1-nano-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    base_config = PretrainedConfig.from_pretrained(model_name)
    qwen3_config = Qwen3Config(base_config,
                               max_lora_adapters=4,
                               max_lora_rank=32,
                               shard_attention_heads=True)
    torch.set_default_device("cuda:0")
    model = Qwen3ForCausalLM(qwen3_config, dtype=base_config.torch_dtype)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    attn_implementation="sdpa",
                                                    use_safetensors=True,
                                                    torch_dtype=base_config.torch_dtype)
    hf_model = PeftModelForCausalLM.from_pretrained(hf_model,
                                                    adapter_name)
    

    for name, param in model.named_parameters():
        if "lora_" in name:        # or use model.peft_config / adapters to filter more cleanly
            print(f"{name}: {param.data}")
            break
    
    
    model_ckpt = "/scratch/yang/.cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
    adapter_ckpt = "/scratch/yang/.cache/hub/models--ethicalabs--Flwr-Qwen3-0.6B-Medical-PEFT/snapshots/faf55d767cdc1f80e7afc7850286746c7ae075b6"
    adapter_ckpt = "/scratch/yang/.cache/hub/models--haizelabs--j1-nano-0.6B/snapshots/a4845eb0e76e46ab13e1dbb752dd6895ff66f210"
    
    load_safetensors_pytorch(
        checkpoint_dir=model_ckpt,
        params_dict=model.state_dict(),
        prefix="",
        filter_fn=None,
        skip_lora=True,
    )

    load_lora_checkpoint(
        model,
        LoraConfig(rank=hf_model.active_peft_config.r,
                   alpha=hf_model.active_peft_config.lora_alpha),
        adapter_index=0,
        checkpoint_path=adapter_ckpt,
    )
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain the theory of relativity in simple terms."}],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = [inputs]

    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    import os

    print("Generating with HuggingFace...")
    print(hf_model)

    os.environ["DO_PRINT"] = "0"
    with torch.no_grad():
        hf_output = hf_model.generate(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    hf_generated_tokens = hf_output.sequences[:, batch.input_ids.shape[1]:].cpu().numpy()
    hf_generated_text = tokenizer.batch_decode(hf_generated_tokens, skip_special_tokens=True)
    print("HuggingFace generated text:", hf_generated_text)

    # Generate with our implementation
    with torch.no_grad():
        generated = model.generation(
            batch.input_ids,
            max_new_tokens=50,
            adapter_indices=0,
        )
    generated_tokens = generated[:, batch.input_ids.shape[1]:].cpu().numpy()
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print("Generated text:", generated_text)

    assert (generated_tokens == hf_generated_tokens).all(), "Generated tokens do not match HuggingFace output."
if __name__ == "__main__":
    test_qwen3_generate()