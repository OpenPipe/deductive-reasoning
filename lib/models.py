from dataclasses import dataclass, field
import torch
from torchtune.models.qwen2_5 import (
    qwen2_5_7b_base,
    qwen2_5_14b_base,
    qwen2_5_14b_instruct,
    qwen2_5_32b_base,
    qwen2_5_32b_instruct,
    qwen2_5_72b_instruct,
)
from torchtune.models.llama3_1 import llama3_1_8b, llama3_1_70b
from torchtune.modules import TransformerDecoder
from typing import Any, Callable, Literal


Optimizer = Literal[
    "torch.optim.AdamW",
    "torchao.prototype.low_bit_optim.AdamW8bit",
    "bitsandbytes.optim.PagedAdamW8bit",
]


@dataclass
class Model:
    """Configuration for language models used in tuning and inference."""

    # Model identity
    base_model: str
    tune_model_type: str
    tune_model: Callable[[], TransformerDecoder]

    # Tuning parameters
    tune_optimizer: Optimizer
    tune_max_batch_tokens: int
    tune_fsdp_cpu_offload: bool = False
    tune_num_output_chunks: int = 8

    # Inference parameters
    vllm_named_arguments: dict[str, Any] = field(default_factory=dict)


def distilled_qwen_7b() -> Model:
    """deepseek-ai/DeepSeek-R1-Distill-Qwen-7B model config."""
    assert torch.cuda.device_count() >= 1, "Qwen-7B requires at least 1 GPU"
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        tune_model_type="QWEN2",
        tune_model=qwen2_5_7b_base,
        tune_optimizer="torch.optim.AdamW",
        tune_max_batch_tokens=32768,
        tune_fsdp_cpu_offload=True,
        vllm_named_arguments={},
    )


def theta_8b() -> Model:
    """NousResearch/Hermes-2-Theta-Llama-3-8B model config."""
    assert torch.cuda.device_count() >= 1, "Llama-8B requires at least 1 GPU"
    return Model(
        base_model="NousResearch/Hermes-2-Theta-Llama-3-8B",
        tune_model_type="LLAMA3",
        tune_model=llama3_1_8b,
        tune_optimizer="torch.optim.AdamW",
        tune_max_batch_tokens=32768,
        tune_fsdp_cpu_offload=True,
        vllm_named_arguments={},
    )


def qwen_14b() -> Model:
    """Qwen/Qwen2.5-14B-Instruct model config."""
    assert torch.cuda.device_count() >= 2, "Qwen-14B requires at least 2 GPUs"
    return Model(
        base_model="Qwen/Qwen2.5-14B-Instruct",
        tune_model_type="QWEN2",
        tune_model=qwen2_5_14b_instruct,
        tune_optimizer="torch.optim.AdamW",
        tune_max_batch_tokens=32768,
        tune_fsdp_cpu_offload=True,
        tune_num_output_chunks=2,
        vllm_named_arguments={},
    )


def distilled_qwen_14b() -> Model:
    """deepseek-ai/DeepSeek-R1-Distill-Qwen-14B model config."""
    assert torch.cuda.device_count() >= 2, "Qwen-14B requires at least 2 GPUs"
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        tune_model_type="QWEN2",
        tune_model=qwen2_5_14b_base,
        tune_optimizer="torch.optim.AdamW",
        tune_max_batch_tokens=32768,
        tune_fsdp_cpu_offload=True,
        tune_num_output_chunks=2,
        vllm_named_arguments={},
    )


def qwen_32b() -> Model:
    """Qwen/Qwen2.5-32B-Instruct model config."""
    assert torch.cuda.device_count() >= 4, "Qwen-32B requires at least 4 GPUs"
    return Model(
        base_model="Qwen/Qwen2.5-32B-Instruct",
        tune_model_type="QWEN2",
        tune_model=qwen2_5_32b_instruct,
        tune_optimizer="torch.optim.AdamW",
        tune_max_batch_tokens=32768,
        tune_fsdp_cpu_offload=True,
        tune_num_output_chunks=2,
        vllm_named_arguments={},
    )


def distilled_qwen_32b() -> Model:
    """deepseek-ai/DeepSeek-R1-Distill-Qwen-32B model config."""
    assert torch.cuda.device_count() >= 4, "Qwen-32B requires at least 4 GPUs"
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        tune_model_type="QWEN2",
        tune_model=qwen2_5_32b_base,
        tune_optimizer="torch.optim.AdamW",
        tune_max_batch_tokens=32768,
        tune_fsdp_cpu_offload=True,
        tune_num_output_chunks=2,
        vllm_named_arguments={},
    )


def llama_70b() -> Model:
    """unsloth/Llama-3.3-70B-Instruct model config."""
    assert torch.cuda.device_count() >= 8, "Llama-70B requires at least 8 GPUs"
    return Model(
        base_model="unsloth/Llama-3.3-70B-Instruct",
        tune_model_type="LLAMA3",
        tune_model=llama3_1_70b,
        tune_optimizer="torch.optim.AdamW",
        tune_max_batch_tokens=32768,
        tune_fsdp_cpu_offload=True,
        tune_num_output_chunks=2,
        vllm_named_arguments={},
    )


def distilled_llama_70b() -> Model:
    """deepseek-ai/DeepSeek-R1-Distill-Llama-70B model config."""
    assert torch.cuda.device_count() >= 8, "Llama-70B requires at least 8 GPUs"
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        tune_model_type="LLAMA3",
        tune_model=llama3_1_70b,
        tune_optimizer="torch.optim.AdamW",
        tune_max_batch_tokens=32768,
        tune_fsdp_cpu_offload=True,
        vllm_named_arguments={},
    )


def qwen_72b() -> Model:
    """Qwen/Qwen2.5-72B-Instruct model config."""
    assert torch.cuda.device_count() >= 8, "Qwen-72B requires at least 8 GPUs"
    return Model(
        base_model="Qwen/Qwen2.5-72B-Instruct",
        tune_model_type="QWEN2",
        tune_model=qwen2_5_72b_instruct,
        tune_optimizer="torch.optim.AdamW",
        tune_max_batch_tokens=32768,
        tune_fsdp_cpu_offload=True,
        tune_num_output_chunks=2,
        vllm_named_arguments={},
    )
