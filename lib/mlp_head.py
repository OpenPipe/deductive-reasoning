import torch
from torch import Tensor
import torch.nn as nn
from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard  # type: ignore
from typing import Any, Optional


class MLPHead(nn.Module):
    """
    MLP head for transformer models that projects the hidden states to scalar values
    for each token position.

    Args:
        hidden_size: Dimension of the transformer's hidden states
        intermediate_size: Size of the intermediate layer (if used)
        dropout_rate: Dropout probability
        use_intermediate_layer: Whether to use an intermediate layer before final projection
        dtype: Data type for the MLP head layers
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        dropout_rate: float = 0.1,
        use_intermediate_layer: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        # Build the MLP head layers
        if use_intermediate_layer:
            if intermediate_size is None:
                intermediate_size = hidden_size // 4

            self.head = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size, dtype=dtype),
                nn.Tanh(),  # Tanh tends to work better than ReLU for value estimation
                nn.Dropout(dropout_rate),
                nn.Linear(intermediate_size, 1, dtype=dtype),
            )
        else:
            self.head = nn.Linear(hidden_size, 1, dtype=dtype)

    def forward(
        self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute predictions for each position in the sequence.

        Args:
            hidden_states: Transformer hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len] where 1 indicates valid tokens
                          and 0 indicates masked tokens. Can be irregular.

        Returns:
            predictions: Predictions [batch_size, seq_len] for each token position
        """
        # Project each position's hidden state to a value
        predictions = self.head(hidden_states).squeeze(-1)  # [batch_size, seq_len]

        # Mask out invalid positions if mask provided
        if attention_mask is not None:
            predictions = predictions * attention_mask

        return predictions

    def materialize_and_shard(
        self, device: torch.device, reshard_after_forward: bool, fsdp_cpu_offload: bool
    ) -> None:
        # Materialize value head parameters before FSDP
        self.head = self.head.to_empty(device=device)

        # Reset parameters
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

        # Apply FSDP to value head
        fsdp_kwargs: dict[str, Any] = {"reshard_after_forward": reshard_after_forward}
        if fsdp_cpu_offload:
            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
        fully_shard(self, **fsdp_kwargs)
