import asyncio
from itertools import cycle, islice
from lib import models
from lib.grpo import GRPO
from lib.inference_early_stop import InferenceEarlyStop
from lib.pack import packed_tensors_from_tokenized_results
from lib.recipe import ComponentConfig, TuneRecipeConfig
from lib.tasks import ChatCompletionParams, get_task_results
from lib.temporal_clue import get_temporal_clue_tasks
from lib.tokenize import TaskResultTokenizer
from lib.tune import (
    clear_iteration_dirs,
    get_iteration,
    get_last_iteration_dir,
    last_tune_log,
    tune,
    Verbosity,
)
from lib.vllm import start_vllm, kill_vllm_workers
import polars as pl
import random
import torch
from transformers import AutoTokenizer
import wandb

run_name = "<YOUR-RUN-NAME>"
assert run_name != "<YOUR-RUN-NAME>", "Don't forget to choose a run name"
run = wandb.init(name=run_name, id=run_name, resume="allow")

# Get tasks
tasks = list(get_temporal_clue_tasks())
val_tasks = tasks[:64]
test_tasks = tasks[64:128]
train_tasks = tasks[128:]
random.seed(42)
random.shuffle(train_tasks)

# GRPO params
wandb.config["clip_epsilon"] = clip_epsilon = 0.2
wandb.config["entropy_coef"] = entropy_coef = 0.0
wandb.config["kl_coef"] = kl_coef = 0.0
wandb.config["tanh"] = tanh = False

# Model params
model = models.qwen_32b()
wandb.config["model"] = model.base_model
tokenizer = AutoTokenizer.from_pretrained(model.base_model)
wandb.config["seq_len"] = seq_len = 16384

# Optimizer params
wandb.config["lr"] = lr = 6e-6
wandb.config["betas"] = betas = (0.9, 0.99)
wandb.config["weight_decay"] = weight_decay = 0.1

# Training params
num_iterations = 1_000
wandb.config["samples_per_task"] = samples_per_task = 50
wandb.config["tasks_per_iter"] = tasks_per_iter = 32
wandb.config["stride"] = stride = 32
output_dir = f"./models/{run_name}"

# Inference params
expected_tokens = 1000  # Initial expected completion tokens per task sample
inference_early_stop = InferenceEarlyStop(alpha=0.992, threshold=-3.0)

# Logging params
verbosity: Verbosity = 2

# Start from the latest iteration if it exists, otherwise start from the base model
model_name = get_last_iteration_dir(output_dir) or model.base_model


async def train() -> None:
    # Loop from the current iteration to the target number of iterations
    for i in range(get_iteration(output_dir), num_iterations):
        # Start vLLM server
        vllm = await start_vllm(
            model_name,
            max_concurrent_requests=4096,
            env={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
            named_arguments=dict(
                block_size=32,
                disable_log_requests=True,
                enable_prefix_caching=True,
                enforce_eager=True,
                gpu_memory_utilization=0.95,
                max_model_len=16384,
                max_num_seqs=4096,
                max_num_batched_tokens=16384,
                num_scheduler_steps=16,
                preemption_mode="swap",
                return_tokens_as_token_ids=True,
                swap_space=80,
                tensor_parallel_size=torch.cuda.device_count(),
            ),
            timeout=360 + 15 * torch.cuda.device_count(),
            verbosity=verbosity,
        )

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(
            int(
                1.3
                * (torch.cuda.device_count() / model.min_gpus)
                * (vllm.max_concurrent_tokens / expected_tokens)
            )
        )

        # Get results for logging validation performance and for tuning with train results
        offset = i * stride
        val_results, train_results = await asyncio.gather(
            get_task_results(
                tasks=val_tasks,
                client=vllm.client,
                model=vllm.model,
                log_results=8,
                n=2,
                on_chunk=inference_early_stop,
                params=ChatCompletionParams(
                    stream_options={
                        "include_usage": True,
                    },
                    max_completion_tokens=8192,
                ),
                pbar_desc="val",
                semaphore=semaphore,
            ),
            get_task_results(
                tasks=list(islice(cycle(train_tasks), offset, offset + tasks_per_iter)),
                client=vllm.client,
                model=vllm.model,
                log_results=False,
                n=samples_per_task,
                on_chunk=inference_early_stop,
                params=ChatCompletionParams(
                    stream_options={
                        "include_usage": True,
                    },
                    max_completion_tokens=8192,
                ),
                pbar_desc="train",
                semaphore=semaphore,
                transform=TaskResultTokenizer(tokenizer),
            ),
        )

        # Stop vLLM workers
        vllm.process.terminate()
        kill_vllm_workers()

        # Log results to Weights & Biases
        val_stats = val_results.stats
        assert val_stats.grades > 0
        assert val_stats.usages > 0
        wandb_data = {
            "iteration": i,
            "exceptions": val_stats.exceptions + train_results.stats.exceptions,
            "reward": val_stats.total_reward / val_stats.grades,
            "tokens": round(val_stats.completion_tokens / val_stats.usages),
        }
        for metric in val_stats.total_metrics:
            wandb_data[metric] = val_stats.total_metrics[metric] / val_stats.grades
        try:
            wandb_data.update(
                pl.DataFrame(last_tune_log(output_dir))
                .drop("step")
                .mean()
                .to_dicts()[0]
            )
        except Exception:
            pass
        wandb.log(wandb_data)

        # Update expected tokens
        expected_tokens = wandb_data["tokens"]

        # Clean up output directory to save space
        try:
            best_iteration = (
                wandb.Api()
                .run(f"{run.entity}/{run.project}/{run.id}")
                .history()
                .sort_values(by="reward")["iteration"]
                .iloc[-1]
            )
            # Clear all but the best and current iterations
            clear_iteration_dirs(output_dir, excluding=[best_iteration, i])
        except Exception:
            pass

        # Pack the tokenized results into tensors
        tokenized_results = [
            result
            for results in train_results
            for result in results
            if result.advantage != 0
        ]
        packed_tensors = packed_tensors_from_tokenized_results(
            tokenized_results,
            seq_len=seq_len,
            pad_token_id=tokenizer.pad_token_id,  # type: ignore
        )
        if verbosity > 0:
            print(f"Packed tensors into {packed_tensors["tokens"].size()} shape")

        # Tune the model
        model_name = await tune(
            base_model=model.base_model if kl_coef > 0 else model_name,
            output_dir=output_dir,
            packed_tensors=packed_tensors,
            model=model.tune_model,
            model_type=model.tune_model_type,
            config=TuneRecipeConfig(
                optimizer=ComponentConfig(
                    "torch.optim.AdamW",
                    lr=lr,
                    betas=betas,
                    weight_decay=weight_decay,
                    fused=True,
                ),
                loss=ComponentConfig(
                    GRPO,
                    clip_epsilon=clip_epsilon,
                    entropy_coef=entropy_coef,
                    kl_coef=kl_coef,
                ),
                shuffle=True,
                batch_size=32768 // seq_len,
                fsdp_cpu_offload=True,
                enable_activation_checkpointing=True,
                enable_activation_offloading=True,
                custom_sharded_layers=["tok_embeddings", "output"],
                num_output_chunks=model.tune_num_output_chunks,
                compile=True,
            ),
            verbosity=verbosity,
        )


asyncio.run(train())
