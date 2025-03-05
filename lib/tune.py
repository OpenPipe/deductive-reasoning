import asyncio
import glob
from lib.mlp_head_checkpointer import MLPHeadCheckpointer
from lib.pack import PackedDataset, PackedTensors, packed_tensors_to_dir
from lib.recipe import ComponentConfig, recipe_main, TuneRecipeConfig
from omegaconf import OmegaConf
import os
import re
import shutil
import sys
import torch
from torchtune.modules import TransformerDecoder
from torchtune.training import cleanup_before_training, FullModelHFCheckpointer
from torchtune.training.metric_logging import DiskLogger
import tqdm
from typing import Any, Callable, Literal, IO


Verbosity = Literal[0, 1, 2]


def clear_iteration_dirs(output_dir: str, excluding: list[int]) -> None:
    for dir in os.listdir(output_dir):
        if (
            os.path.isdir(os.path.join(output_dir, dir))
            and dir.isdigit()
            and int(dir) not in excluding
        ):
            iteration_dir = os.path.join(output_dir, dir)
            chat_logs_dir = os.path.join(iteration_dir, "chat-completion-logs")

            if os.path.isdir(chat_logs_dir):
                # Save the chat-completion-logs directory
                temp_dir = os.path.join(output_dir, f"temp-{dir}-logs")
                shutil.move(chat_logs_dir, temp_dir)

                # Delete the iteration directory
                shutil.rmtree(iteration_dir)

                # Recreate the iteration directory and move logs back
                os.makedirs(iteration_dir)
                shutil.move(temp_dir, chat_logs_dir)
                print(
                    f"Cleared iteration directory {iteration_dir} except chat-completion-logs"
                )
            else:
                # No chat logs, delete the entire directory
                shutil.rmtree(iteration_dir)
                print(f"Deleted iteration directory {iteration_dir}")


def get_iteration(output_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    return max(
        (
            int(subdir)
            for subdir in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, subdir)) and subdir.isdigit()
        ),
        default=0,
    )


def get_last_iteration_dir(output_dir: str) -> str | None:
    last_iteration_dir = os.path.join(output_dir, f"{get_iteration(output_dir):04d}")
    return last_iteration_dir if os.path.exists(last_iteration_dir) else None


def last_tune_log(output_dir: str) -> list[dict[str, float]]:
    sorted_logs = sorted(glob.glob(f"{output_dir}/logs/*"))
    contents = open(sorted_logs[-1]).read()
    lines = contents.strip().splitlines()
    parsed_logs = []
    for line in lines:
        step_part, metrics_part = line.split(" | ")
        step = int(step_part.split()[1])
        metrics = {}
        for metric in metrics_part.split():
            key, value = metric.split(":")
            metrics[key] = float(value)
        parsed_logs.append({"step": step, **metrics})
    return parsed_logs


async def tune(
    base_model: str,
    output_dir: str,
    packed_tensors: PackedTensors,
    model: Callable[[], TransformerDecoder],
    model_type: str,
    config: TuneRecipeConfig = TuneRecipeConfig(),
    in_process: bool = False,
    verbosity: Verbosity = 2,
) -> str:
    if os.path.isdir(base_model):
        base_checkpoint_dir = base_model
    else:
        process = await asyncio.create_subprocess_shell(
            f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {base_model}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        base_stdout = []

        async def read_stream(stream, is_stderr=False):
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode().rstrip()
                if is_stderr:
                    print(text)
                else:
                    base_stdout.append(text)

        await asyncio.gather(
            read_stream(process.stdout, is_stderr=False),
            read_stream(process.stderr, is_stderr=True),
        )
        await process.wait()
        base_checkpoint_dir = "\n".join(base_stdout).strip()

    config.checkpointer = _get_checkpointer_config(
        checkpoint_dir=max(
            (d for d in glob.glob(f"{output_dir}/*") if d.split("/")[-1].isdigit()),
            key=lambda x: int(x.split("/")[-1]),
            default=base_checkpoint_dir,
        ),
        output_dir=output_dir,
        tune_model_type=model_type,
    )
    if config.loss.kl_coef > 0:
        print("Using reference checkpointer")
        config.reference_checkpointer = _get_checkpointer_config(
            checkpoint_dir=base_checkpoint_dir,
            output_dir=output_dir,
            tune_model_type=model_type,
        )
    if config.metric_logger is None:
        config.metric_logger = ComponentConfig(DiskLogger, log_dir=f"{output_dir}/logs")
    config.model = ComponentConfig(model)
    disk_packed_tensors = packed_tensors_to_dir(packed_tensors, f"{output_dir}/tensors")
    config.dataset = ComponentConfig(
        PackedDataset,
        **disk_packed_tensors,
    )
    config.seed = 42
    dict_config = config.dict_config()
    OmegaConf.save(dict_config, f"{output_dir}/config.yaml")
    if in_process:
        cleanup_before_training()
        recipe_main(config)
    else:
        await _tune_run(
            config_path=f"{output_dir}/config.yaml",
            total=disk_packed_tensors["num_sequences"],
            verbosity=verbosity,
            torchrun_kwargs={"nproc_per_node": torch.cuda.device_count()},
            # tune_run_env={"CUDA_LAUNCH_BLOCKING": "1"},
        )
    epoch_dirs = lambda: glob.glob(f"{output_dir}/epoch_*")
    epoch_dir = max(
        epoch_dirs(),
        key=lambda x: int(x.split("_")[-1]),
        default=None,
    )
    assert (
        epoch_dir is not None
    ), f"No epoch directory found in output directory {output_dir}"
    iteration_dir = f"{output_dir}/{get_iteration(output_dir) + 1:04d}"
    os.rename(epoch_dir, iteration_dir)
    for epoch_dir in epoch_dirs():
        os.rmdir(epoch_dir)
    return iteration_dir


def _get_checkpointer_config(
    checkpoint_dir: str,
    output_dir: str,
    tune_model_type: str,
    checkpoint_files: list[str] | None = None,
    mlp_head_checkpointer: bool = False,
    output_subdir: str = "",
) -> ComponentConfig[FullModelHFCheckpointer]:
    return ComponentConfig(
        MLPHeadCheckpointer if mlp_head_checkpointer else FullModelHFCheckpointer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_files=checkpoint_files
        or [
            os.path.basename(file)
            for ext in ["safetensors", "pt", "ckpt", "bin", "pth"]
            for file in glob.glob(f"{checkpoint_dir}/*.{ext}")
            if not file.endswith("mlp_head.pt")
        ],
        recipe_checkpoint=None,
        output_dir=output_dir + output_subdir,
        model_type=tune_model_type,
    )


async def _tune_run(
    config_path: str,
    total: int,
    verbosity: Verbosity = 2,
    torchrun_kwargs: dict[str, Any] | None = None,
    tune_run_env: dict[str, str] | None = None,
) -> None:
    args = [
        "tune",
        "run",
        *[
            f"--{key.replace('_', '-')}{f'={value}' if value is not True else ''}"
            for key, value in (torchrun_kwargs or {}).items()
        ],
        "lib.recipe.TuneRecipe",
        "--config",
        config_path,
    ]
    if verbosity > 0:
        print(f"$ {' '.join(args)}")
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **(tune_run_env or {})},
    )
    if verbosity == 1:
        pbar = tqdm.tqdm(total=total)
    else:
        pbar = None

    async def log_output(stream: asyncio.StreamReader, io: IO[str]) -> None:
        output = ""
        while True:
            try:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                output += chunk.decode()
                if verbosity > 1:
                    io.write(output)
                    io.flush()
                    output = ""
                elif verbosity == 1:
                    output = output.split("\n")[-1]
                    if pbar:
                        pbar_start = re.compile(r"(\d+)\|(\d+)\|Loss: ([\d.]+):")
                        if match := pbar_start.search(output):
                            epoch, step, loss = match.groups()
                            pbar.update(int(step) - pbar.n)
                            pbar.set_description(f"{epoch}|{step}|Loss: {loss}")
                        metrics = {
                            key: value
                            for key, value in re.findall(r"(\w+)=([\d.-]+)", output)
                        }
                        if metrics:
                            pbar.set_postfix(**metrics)
                            output = ""
                    else:
                        pbar_regex = re.compile(
                            r"\[(?:\d+:)?\d+:\d+<(?:\d+:)?\d+:\d+.*\]"
                        )
                        if pbar_regex.search(output):
                            io.write(output)
                            io.flush()
                            output = ""
            except Exception:
                break

    tasks = []
    if process.stdout:
        tasks.append(asyncio.create_task(log_output(process.stdout, sys.stdout)))
    if process.stderr:
        tasks.append(asyncio.create_task(log_output(process.stderr, sys.stderr)))
    try:
        _ = await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        process.kill()
    if pbar:
        pbar.close()


def _save_last_checkpoint_files(base_checkpoint_dir: str, output_dir: str) -> str:
    """
    Saves and returns the directory of the latest checkpoint.
    """
    # Find the latest epoch number from model checkpoint files
    epoch = max(
        (
            int(result.group(1))
            for result in (
                re.search(r"hf_model_\d+_(\d+)\.pt", file)
                for file in glob.glob(f"{output_dir}/hf_model_*_*.pt")
            )
            if result
        ),
        default=None,
    )

    assert (
        epoch is not None
    ), f"No model checkpoint files found to save in output directory {output_dir}"

    iteration, iteration_dir = _create_iteration_dir(base_checkpoint_dir, output_dir)

    # Move model checkpoint files to the iteration directory
    for src in [
        path
        for extension in ("pt", "pt.ignore")
        for path in glob.glob(f"{output_dir}/*_{epoch}.{extension}")
    ]:
        dst = f"{iteration_dir}/{os.path.basename(src).replace(f'_{epoch}.pt', '.pt')}"
        shutil.move(src, dst)

    # Delete all checkpoint files in the output directory
    for file in [
        path
        for extension in ("pt", "pt.ignore")
        for path in glob.glob(f"{output_dir}/*_*.{extension}")
    ]:
        os.remove(file)

    print(f"Saved iteration #{iteration} model files to {iteration_dir}")
    return iteration_dir


def _create_iteration_dir(base_checkpoint_dir: str, output_dir: str) -> tuple[int, str]:
    next_iteration = get_iteration(output_dir) + 1

    # Create a new directory for this iteration
    iteration_dir = f"{output_dir}/{next_iteration:04d}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(iteration_dir, exist_ok=False)

    # Copy configuration files (non-model files) to the iteration directory
    for file in os.listdir(base_checkpoint_dir):
        if not any(
            file.endswith(suffix)
            for suffix in (".safetensors", ".pt", ".ckpt", ".bin", ".pth", ".h5")
        ):
            src = os.path.join(base_checkpoint_dir, file)
            dst = os.path.join(iteration_dir, file)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    return next_iteration, iteration_dir
