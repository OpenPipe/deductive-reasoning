from pathlib import Path
import torch
from torchtune.training import MODEL_KEY, ADAPTER_KEY, ADAPTER_CONFIG
from torchtune.training.checkpointing import FullModelHFCheckpointer
from torchtune.training.checkpointing._utils import safe_torch_load
from torchtune.utils import get_logger
from typing import Dict, Any

logger = get_logger("DEBUG")

# Define constant for MLP head key
MLP_HEAD_KEY = "mlp_head"


class MLPHeadCheckpointer(FullModelHFCheckpointer):
    """
    Extends FullModelHFCheckpointer to support loading and saving MLP head weights.
    MLP head weights are stored in a separate file to maintain compatibility with
    base model checkpoints.
    """

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load checkpoint including MLP head weights if present.
        """
        # First load the base model checkpoint
        converted_state_dict = super().load_checkpoint()

        # Try to load MLP head weights if they exist
        mlp_head_path = self._checkpoint_dir / "mlp_head.pt.ignore"
        if mlp_head_path.exists():
            logger.info(f"Loading MLP head weights from {mlp_head_path}")
            converted_state_dict[MLP_HEAD_KEY] = safe_torch_load(mlp_head_path)
        else:
            logger.info("No MLP head weights found, will initialize from scratch")

        return converted_state_dict

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
    ) -> None:
        """
        Save checkpoint with MLP head weights in a separate file.

        Args:
            state_dict: Dictionary containing model state and optionally MLP head weights
            epoch: Current epoch number
            intermediate_checkpoint: Whether this is a mid-training checkpoint
            adapter_only: Whether to only save adapter weights
        """
        # Save MLP head weights if present
        if MLP_HEAD_KEY in state_dict:
            output_path = Path.joinpath(
                self._output_dir, f"mlp_head_{epoch}"
            ).with_suffix(".pt.ignore")
            torch.save(state_dict[MLP_HEAD_KEY], output_path)
            logger.info(f"MLP head checkpoint saved to {output_path}")

        # Remove MLP head from state dict before calling parent method
        # to maintain compatibility with base checkpointer
        mlp_head_state = state_dict.pop(MLP_HEAD_KEY, None)

        # Save the rest of the checkpoint using parent implementation
        super().save_checkpoint(
            state_dict, epoch, intermediate_checkpoint, adapter_only
        )

        # Restore MLP head state if we removed it
        if mlp_head_state is not None:
            state_dict[MLP_HEAD_KEY] = mlp_head_state
