import torch
from torch.nn import functional as F, init

class LoRAMixin:

    def init_lora(
            self,
            *,
            max_lora_adapters: int,
            max_lora_rank: int,
            shape_A: tuple[int, ...], # (max_lora_adapters, in_features, max_lora_rank)
            shape_B: tuple[int, ...], # (max_lora_adapters, max_lora_rank, out_features)
            dtype: torch.dtype,
    ) -> None:
        self.max_lora_adapters = max_lora_adapters
        self.max_lora_rank = max_lora_rank

        if max_lora_adapters <= 0:
            self.lora_scaling = None
            self.lora_ranks = None
            self.lora_A = None
            self.lora_B = None
        else:
            self.register_buffer(
                'lora_scaling',
                torch.ones(
                    (max_lora_adapters,),
                    dtype=dtype,
                )
            )
            self.register_buffer(
                'lora_ranks',
                torch.ones(
                    (max_lora_adapters,),
                    dtype=torch.int32,
                ) * max_lora_rank
            )

            self.lora_A = torch.nn.Parameter(
                torch.empty(
                    shape_A,
                    dtype=dtype,
                )
            )
            self.lora_B = torch.nn.Parameter(
                torch.empty(
                    shape_B,
                    dtype=dtype,
                )
            )
            init.kaiming_normal_(self.lora_A)
            init.zeros_(self.lora_B)
    

    def apply_lora(
            self,
            x: torch.Tensor,
            base_output: torch.Tensor,
            adapter_indices: torch.Tensor | None,
            ) -> torch.Tensor:
        if self.max_lora_adapters <= 0 or adapter_indices is None:
            return base_output
        
        if not hasattr(self, 'lora_A') or not hasattr(self, 'lora_B'):
            raise AttributeError("LoRA parameters not initialized")
        
        (batch_size, seq_len, hidden_dim) = x.shape
        assert len(self.lora_A.shape) == 3
        assert hidden_dim == self.lora_A.shape[2], f"Input hidden dim {hidden_dim} does not match LoRA A shape {self.lora_A.shape}"
        assert adapter_indices.shape[0] == batch_size, f"Adapter indices batch size {adapter_indices.shape[0]} does not match input batch size {batch_size}"

        lora_outs = torch.zeros_like(base_output)
        for i in range(batch_size):
            adapter_idx = adapter_indices[i].item()
            rank = self.lora_ranks[adapter_idx].item()
            
            A = self.lora_A[adapter_idx]
            B = self.lora_B[adapter_idx]
            intermediate = torch.matmul(x[i], A.T)
            lora_out = torch.matmul(intermediate, B.T)
            lora_outs[i] = lora_out * self.lora_scaling[adapter_idx]
            # print(f"adapter_idx: {adapter_idx}, rank: {rank}, adapter_B norm: {B.norm().item():.6f}")
            # print(f"LoRA applied for batch {i}, adapter {adapter_idx}, rank {rank}, delta norm {lora_out.norm().item():.6f}")
        return base_output + lora_outs
    

class LoRALinear(torch.nn.Linear, LoRAMixin):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            dtype: torch.dtype = torch.float32,
            max_lora_adapters: int = 0,
            max_lora_rank: int = 0,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.init_lora(
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            shape_A=(max_lora_adapters, max_lora_rank, in_features, ), # this is transpose of jax version but matches the peft implementation
            shape_B=(max_lora_adapters, out_features, max_lora_rank, ), # this is transpose of jax version but matches the peft implementation
            dtype=dtype,
        )
    
    def forward(
            self,
            input: torch.Tensor,
            adapter_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_output = super().forward(input)
        return self.apply_lora(input, base_output, adapter_indices)