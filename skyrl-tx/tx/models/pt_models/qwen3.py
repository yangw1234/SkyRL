import torch
from torch import nn
from tx.models.configs import Qwen3Config
from tx.layers.pt_layers.lora import LoRALinear
from transformers.integrations.sdpa_attention import sdpa_attention_forward

class RMSNorm(torch.nn.Module):

    def __init__(self, size: int, *, eps:float = 1e-6, dtype: torch.dtype):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(size, dtype=dtype))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


def apply_rope(inputs: torch.Tensor, position_ids: torch.Tensor, head_dim: int, theta: int) -> torch.Tensor:

    # fraction = 2 * torch.arange(0, head_dim//2, device=inputs.device, dtype=torch.float) / head_dim
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=inputs.device, dtype=torch.float) / head_dim)
    x = (position_ids[..., None].float() * inv_freq[None, None, :])[..., None, :]
    sin, cos = x.sin().to(inputs.dtype), x.cos().to(inputs.dtype)
    a, b = inputs[..., :head_dim//2], inputs[..., head_dim//2:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1).to(inputs.dtype)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Qwen3Attention(torch.nn.Module):

    def __init__(self, config: Qwen3Config, * , dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = LoRALinear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            bias=config.attention_bias,
            dtype=dtype,)
        
        self.k_proj = LoRALinear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            bias=config.attention_bias,
            dtype=dtype)
        
        self.v_proj = LoRALinear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            bias=config.attention_bias,
            dtype=dtype)
        
        self.o_proj = LoRALinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            bias=config.attention_bias,
            dtype=dtype)
        
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)

    def forward(
            self,
            hidden_states: torch.Tensor,
            *,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            adapter_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        query = self.q_norm(
            self.q_proj(hidden_states, adapter_indices=adapter_indices).reshape(batch_size,
                                               seq_len,
                                               self.num_heads,
                                               self.head_dim)
        )
        key = self.k_norm(
            self.k_proj(hidden_states, adapter_indices=adapter_indices).reshape(batch_size,
                                               seq_len,
                                               self.num_key_value_heads,
                                               self.head_dim)
        )
        value = self.v_proj(hidden_states, adapter_indices=adapter_indices).reshape(batch_size,
                                                     seq_len,
                                                     self.num_key_value_heads,
                                                     self.head_dim)
        query = apply_rope(query, position_ids, self.head_dim, self.config.rope_theta)
        key = apply_rope(key, position_ids, self.head_dim, self.config.rope_theta)

        query = query.transpose(1, 2).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        key = key.transpose(1, 2).reshape(batch_size, self.num_key_value_heads, seq_len, self.head_dim)
        value = value.transpose(1, 2).reshape(batch_size, self.num_key_value_heads, seq_len, self.head_dim)
        # print(f"Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")

        # repeat_kv_times = self.num_heads // self.num_key_value_heads
        # key = repeat_kv(key, repeat_kv_times)
        # value = repeat_kv(value, repeat_kv_times)

        scaling = 1.0 / (self.head_dim ** 0.5)
        attn_output, _ = sdpa_attention_forward(
            self,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=0.0,
            is_causal=True,
        )
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # print(f"Attention output shape: {attn_output.shape}")

        return self.o_proj(attn_output, adapter_indices=adapter_indices)

class Qwen3MLP(torch.nn.Module):

    def __init__(self, config: Qwen3Config, *, dtype: torch.dtype):
        super().__init__()
        self.gate_proj = LoRALinear(config.hidden_size,
                                    config.intermediate_size,
                                    max_lora_adapters=config.max_lora_adapters,
                                    max_lora_rank=config.max_lora_rank,
                                    bias=False,
                                    dtype=dtype)
        self.up_proj = LoRALinear(config.hidden_size,
                                  config.intermediate_size,
                                  max_lora_adapters=config.max_lora_adapters,
                                  max_lora_rank=config.max_lora_rank,
                                  bias=False,
                                  dtype=dtype)
        self.down_proj = LoRALinear(config.intermediate_size,
                                    config.hidden_size,
                                    max_lora_adapters=config.max_lora_adapters,
                                    max_lora_rank=config.max_lora_rank,
                                    bias=False,
                                    dtype=dtype)
        self.act_fn = nn.functional.silu

    def forward(self, hidden_states: torch.Tensor, adapter_indices: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.act_fn(self.gate_proj(hidden_states, adapter_indices=adapter_indices)) * self.up_proj(hidden_states, adapter_indices=adapter_indices)
        hidden_states = self.down_proj(hidden_states, adapter_indices=adapter_indices)
        return hidden_states

class Qwen3DecoderLayer(torch.nn.Module):

    def __init__(self, config: Qwen3Config, *, dtype: torch.dtype):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.self_attn = Qwen3Attention(config, dtype=dtype)
        self.mlp = Qwen3MLP(config, dtype=dtype)

    def forward(
            self,
            hidden_states: torch.Tensor,
            *,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            adapter_indices: torch.Tensor | None = None,
            do_print: bool = False,
    ) -> torch.Tensor:
        if do_print:
            print(f"Input to layernorm: {hidden_states}")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if do_print:
            print(f"After layernorm: {hidden_states}")
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            adapter_indices=adapter_indices,
        )
        if do_print:
            print(f"After self attention: {hidden_states}")
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if do_print:
            print(f"After post-attention layernorm: {hidden_states}")
        hidden_states = self.mlp(hidden_states, adapter_indices=adapter_indices)
        if do_print:
            print(f"After MLP: {hidden_states}")
        hidden_states = residual + hidden_states

        return hidden_states

class Qwen3Model(torch.nn.Module):

    def __init__(self, config: Qwen3Config, *, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = torch.nn.ModuleList(
            [Qwen3DecoderLayer(config, dtype=dtype) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def forward(
            self,
            input_ids: torch.Tensor,
            *,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            adapter_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        import os
        do_print = "DO_PRINT" in os.environ and os.environ["DO_PRINT"] == "1"
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                adapter_indices=adapter_indices,
                do_print=(i==0) if do_print else False,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

class Qwen3ForCausalLM(torch.nn.Module):

    def __init__(self, config: Qwen3Config, *, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, dtype=dtype)
        # if not config.tie_word_embeddings:
        self.lm_head = LoRALinear(config.hidden_size,
                                  config.vocab_size,
                                  max_lora_adapters=config.max_lora_adapters,
                                  max_lora_rank=config.max_lora_rank,
                                  bias=False, dtype=dtype)
    
    def forward(
            self,
            input_ids: torch.Tensor,
            *,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            adapter_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            adapter_indices=adapter_indices,
        )
        # if self.config.tie_word_embeddings:
        #     lm_logits = torch.matmul(hidden_states, self.model.embed_tokens.weight.t())
        # else:
        lm_logits = self.lm_head(hidden_states, adapter_indices=adapter_indices)
        return lm_logits
    

    def generation(self,
                   input_ids: torch.Tensor,
                   max_new_tokens: int,
                   adapter_indices: int) -> torch.Tensor:

        batch = input_ids.shape[0]
        assert batch == 1, "Only batch size of 1 is supported for generation."
        seq_len = input_ids.shape[1]
        generated = input_ids

        for i in range(max_new_tokens):
            attention_mask = None
            position_ids = torch.arange(
                seq_len + i,
                device=input_ids.device,
                dtype=input_ids.dtype,
            ).unsqueeze(0)
            adapter_indices_tensor = torch.full(
                (1,), adapter_indices,
                device=input_ids.device,
                dtype=torch.long,
            )
            logits = self.forward(
                generated,
                attention_mask=attention_mask,
                position_ids=position_ids,
                adapter_indices=adapter_indices_tensor,
            )
            next_token_ids = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token_ids), dim=1)

        return generated
