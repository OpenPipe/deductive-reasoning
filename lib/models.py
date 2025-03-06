from dataclasses import dataclass
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
from typing import Callable


@dataclass
class Model:
    """Basic language model configuration"""

    base_model: str
    min_gpus: int
    tune_model_type: str
    tune_model: Callable[[], TransformerDecoder]
    tune_num_output_chunks: int

    def __post_init__(self) -> None:
        assert (
            torch.cuda.device_count() >= self.min_gpus
        ), f"{self.base_model} requires at least {self.min_gpus} GPUs"


def distilled_qwen_7b() -> Model:
    """deepseek-ai/DeepSeek-R1-Distill-Qwen-7B model config."""
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        min_gpus=1,
        tune_model_type="QWEN2",
        tune_model=qwen2_5_7b_base,
        tune_num_output_chunks=8,
    )


def theta_8b() -> Model:
    """NousResearch/Hermes-2-Theta-Llama-3-8B model config."""
    return Model(
        base_model="NousResearch/Hermes-2-Theta-Llama-3-8B",
        min_gpus=1,
        tune_model_type="LLAMA3",
        tune_model=llama3_1_8b,
        tune_num_output_chunks=8,
    )


def qwen_14b() -> Model:
    """Qwen/Qwen2.5-14B-Instruct model config."""
    return Model(
        base_model="Qwen/Qwen2.5-14B-Instruct",
        min_gpus=2,
        tune_model_type="QWEN2",
        tune_model=qwen2_5_14b_instruct,
        tune_num_output_chunks=2,
    )


def distilled_qwen_14b() -> Model:
    """deepseek-ai/DeepSeek-R1-Distill-Qwen-14B model config."""
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        min_gpus=2,
        tune_model_type="QWEN2",
        tune_model=qwen2_5_14b_base,
        tune_num_output_chunks=2,
    )


def qwen_32b() -> Model:
    """Qwen/Qwen2.5-32B-Instruct model config."""
    return Model(
        base_model="Qwen/Qwen2.5-32B-Instruct",
        min_gpus=4,
        tune_model_type="QWEN2",
        tune_model=qwen2_5_32b_instruct,
        tune_num_output_chunks=2,
    )


def distilled_qwen_32b() -> Model:
    """deepseek-ai/DeepSeek-R1-Distill-Qwen-32B model config."""
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        min_gpus=4,
        tune_model_type="QWEN2",
        tune_model=qwen2_5_32b_base,
        tune_num_output_chunks=2,
    )


def llama_70b() -> Model:
    """unsloth/Llama-3.3-70B-Instruct model config."""
    return Model(
        base_model="unsloth/Llama-3.3-70B-Instruct",
        min_gpus=8,
        tune_model_type="LLAMA3",
        tune_model=llama3_1_70b,
        tune_num_output_chunks=2,
    )


def distilled_llama_70b() -> Model:
    """deepseek-ai/DeepSeek-R1-Distill-Llama-70B model config."""
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        min_gpus=8,
        tune_model_type="LLAMA3",
        tune_model=llama3_1_70b,
        tune_num_output_chunks=8,
    )


def qwen_72b() -> Model:
    """Qwen/Qwen2.5-72B-Instruct model config."""
    return Model(
        base_model="Qwen/Qwen2.5-72B-Instruct",
        min_gpus=8,
        tune_model_type="QWEN2",
        tune_model=qwen2_5_72b_instruct,
        tune_num_output_chunks=2,
    )
