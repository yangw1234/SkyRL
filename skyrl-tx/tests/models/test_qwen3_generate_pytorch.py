from transformers import AutoTokenizer, PretrainedConfig, AutoModelForCausalLM
from tx.models.configs import Qwen3Config
from tx.models.pt_models.qwen3 import Qwen3ForCausalLM
from tx.utils.models_torch import load_safetensors_pytorch
import torch
def test_qwen3_generate():
    """Test batched text generation with KV caching matches HuggingFace."""
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    base_config = PretrainedConfig.from_pretrained(model_name)
    qwen3_config = Qwen3Config(base_config,
                               max_lora_adapters=32,
                               max_lora_rank=32,
                               shard_attention_heads=True)
    torch.set_default_device("cuda:3")
    model = Qwen3ForCausalLM(qwen3_config, dtype=base_config.torch_dtype)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    attn_implementation="sdpa",
                                                    use_safetensors=True,
                                                    torch_dtype=base_config.torch_dtype)

    load_safetensors_pytorch(
        checkpoint_dir="/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
        config=qwen3_config,
        model=model,
        skip_lora=True,
        prefix="",
        filter_fn=None,
    )
    inputs = ["The capital of France is"]

    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    import os

    os.environ["DO_PRINT"] = "0"
    with torch.no_grad():
        hf_output = hf_model.generate(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            max_new_tokens=20,
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
            max_new_tokens=20,
        )
    generated_tokens = generated[:, batch.input_ids.shape[1]:].cpu().numpy()
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print("Generated text:", generated_text)

    assert (generated_tokens == hf_generated_tokens).all(), "Generated tokens do not match HuggingFace output."
if __name__ == "__main__":
    test_qwen3_generate()