from aioitertools.helpers import maybe_await
import asyncio
from dataclasses import dataclass, field
import numpy as np
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.completion_usage import CompletionUsage
import random
from typing import Awaitable, Callable, TypeVar

from .chat_completions import get_chat_completion, TokenScheduler
from .types import ChatCompletionParams
from .tqdm import tqdm

# A grader function returns a floating-point reward,
# and optionally a dictionary of metrics as the second return value
Grade = float | tuple[float, dict[str, float]]
Grader = Callable[[Choice], Grade | Awaitable[Grade]]


@dataclass
class Task:
    """
    A minimal task definition.

    Args:
        messages (list[ChatCompletionMessageParam]): OpenAI API compatible chat messages for prompting the model
        grader (Grader): A grader function to score the model's responses
    """

    messages: list[ChatCompletionMessageParam]
    grader: Grader


@dataclass
class TaskResult:
    """
    A single task result.

    Args:
        task (Task): The task that was graded
        chat_completions (list[ChatCompletion]): The chat completions generated for the task
        rewards (dict[tuple[str, int], float]): Rewards for each chat completion and choice index
        metrics (dict[tuple[str, int], dict[str, float]]): Metrics for each chat completion and choice index
        advantages (dict[tuple[str, int], float]): GRPO advantages for each chat completion and choice index
        exceptions (list[Exception]): Exceptions that occurred while getting the task result
    """

    task: Task
    chat_completions: list[ChatCompletion]
    rewards: dict[tuple[str, int], float]
    metrics: dict[tuple[str, int], dict[str, float]]
    advantages: dict[tuple[str, int], float]
    exceptions: list[Exception]


@dataclass
class TaskResultStats:
    """
    Statistics for task results.

    Args:
        pbar (tqdm.tqdm): The progress bar
        prices (tuple[float, float] | None): Prices for input/output tokens
        completion_tokens (int): Total completion tokens
        exceptions (list[Exception]): Exceptions that occurred while getting the task results
        grades (int): Number of grades
        new_completion_ids (set[str]): Set of new completion IDs
        new_completion_tokens (int): Total new completion tokens
        new_prompt_tokens (int): Total new prompt tokens
        prompt_tokens (int): Total prompt tokens
        token_logprobs (int): Total token log probabilities
        total_metrics (dict[str, float]): Total metrics
        total_reward (float): Total reward
        usages (int): Total usages
    """

    pbar: tqdm.tqdm
    prices: tuple[float, float] | None
    completion_tokens: int = 0
    exceptions: list[Exception] = field(default_factory=list)
    grades: int = 0
    new_completion_ids: set[str] = field(default_factory=set)
    new_completion_tokens: int = 0
    new_prompt_tokens: int = 0
    prompt_tokens: int = 0
    token_logprobs: int = 0
    total_metrics: dict[str, float] = field(default_factory=dict)
    total_reward: float = 0
    usages: int = 0

    def __del__(self) -> None:
        self.pbar.close()

    def update(
        self,
        *,
        id: str | None,
        chunk: ChatCompletionChunk | None = None,
        usage: CompletionUsage | None = None,
        reward: float | None = None,
        metrics: dict[str, float] | None = None,
        exception: Exception | None = None,
    ) -> None:
        if chunk:
            if id is not None:
                self.new_completion_ids.add(id)
            self.token_logprobs += sum(
                len(choice.logprobs.content or choice.logprobs.refusal or [])
                for choice in chunk.choices
                if choice.logprobs
            )
        elif usage:
            self.completion_tokens += usage.completion_tokens
            self.prompt_tokens += usage.prompt_tokens
            self.usages += 1
            if id in self.new_completion_ids:
                self.new_completion_tokens += usage.completion_tokens
                self.new_prompt_tokens += usage.prompt_tokens
        elif reward is not None:
            self.grades += 1
            self.total_reward += reward
            self.pbar.update()
            if metrics:
                for key, value in metrics.items():
                    if key not in self.total_metrics:
                        self.total_metrics[key] = 0
                    self.total_metrics[key] += value
        elif exception:
            self.exceptions.append(exception)
        postfix = {
            "completion_tokens": round(self.completion_tokens / max(self.usages, 1)),
            "prompt_tokens": round(self.prompt_tokens / max(self.usages, 1)),
            "reward": self.total_reward / max(self.grades, 1),
        }
        for key, value in self.total_metrics.items():
            postfix[key] = value / max(self.grades, 1)
        if self.prices:
            postfix["spend"] = (
                f"${(
                self.new_prompt_tokens * self.prices[0]
                + (self.token_logprobs or self.new_completion_tokens) * self.prices[1]
            ) / 1_000_000:.2f}"
            )
        if self.token_logprobs:
            postfix["token_logprobs"] = self.token_logprobs
        if self.exceptions:
            postfix["exceptions"] = len(self.exceptions)
        self.pbar.set_postfix(postfix)


T = TypeVar("T")


class TaskResults(list[T]):
    stats: TaskResultStats


async def get_task_results(
    tasks: list[Task],
    client: AsyncOpenAI,
    model: str,
    cache: bool = True,
    clear_pbar: bool = False,
    print_pbar: bool = True,
    log_dir: str | None = None,
    log_results: bool | float | int = True,
    log_token_logprobs: bool = True,
    n: int = 1,
    on_chunk: Callable[[ChatCompletionChunk, ChatCompletion], None] | None = None,
    params: ChatCompletionParams | None = None,
    pbar_desc: str | None = None,
    prices: tuple[float, float] | None = None,
    semaphore: asyncio.Semaphore | None = None,
    token_scheduler: TokenScheduler | None = None,
    transform: Callable[[TaskResult], T | Awaitable[T]] = lambda x: x,
) -> TaskResults[T]:
    """
    Returns results for tasks using an AsyncOpenAI client for a given model. Includes support for caching, rate limiting, and logging. Results may be optionally transformed.

    Args:
        tasks (list[Task]): List of Task objects, each containing messages to send to the LLM and a grader function
        client (AsyncOpenAI): Any valid AsyncOpenAI client that supports creating chat completions (may be pointed at API providers other than OpenAI or a local inference engine like vLLM)
        model (str): Name of the chat completion model to use
        cache (bool): Whether to cache completion results for later reuse
        clear_pbar (bool): Whether to clear the progress bar after completion
        print_pbar (bool): Whether to print the progress bar summary after completion
        log_dir (str | None): Directory to save completion logs to. If None, will use the default chat completion log directory
        log_results (bool | float | int): Controls which task results to log. Can be a boolean, float (fraction), or int (count)
        log_token_logprobs (bool): Whether to stream token log probabilities count to the progress bar
        n (int): Number of chat completions to sample per task
        on_chunk (Callable[[ChatCompletionChunk, ChatCompletion], None] | None): Optional callback function for processing completion chunks
        params (ChatCompletionParams | None): Additional parameters to pass to the chat completion API
        pbar_desc (str | None): Description to display on the progress bar
        prices (tuple[float, float] | None): Tuple of (input_price, output_price) per million tokens, for cost tracking
        semaphore (asyncio.Semaphore | None): Optional semaphore for limiting concurrent API calls
        token_scheduler (TokenScheduler | None): Optional token scheduler for rate limiting
        transform (Callable[[TaskResult], T | Awaitable[T]]): Function to transform TaskResult objects before returning

    Returns:
        TaskResults[T]: Processed results and statistics

    Process: Runs model inference → evaluates with graders → computes rewards/advantages →
    tracks metrics → transforms results
    """
    num_completions = len(tasks) * n
    pbar = tqdm.tqdm(total=num_completions, desc=pbar_desc)
    stats = TaskResultStats(pbar=pbar, prices=prices)
    results = TaskResults(
        await asyncio.gather(
            *(
                _get_task_result(
                    task=task,
                    client=client,
                    model=model,
                    log_results=log_results,
                    n=n,
                    cache=cache,
                    log_dir=log_dir,
                    on_chunk=_create_on_chunk_callback(
                        log_token_logprobs, on_chunk, stats
                    ),
                    semaphore=semaphore,
                    token_scheduler=token_scheduler,
                    params=params,
                    stats=stats,
                    transform=transform,
                )
                for task, log_results in zip(
                    tasks, _get_log_results_flags(log_results, len(tasks))
                )
            )
        )
    )
    results.stats = stats
    pbar.close()
    if getattr(pbar, "container", None) and clear_pbar:
        pbar.container.close()
    if getattr(pbar, "container", None) and print_pbar:
        print(pbar.container.__repr__(pretty=True))
    return results


def _create_on_chunk_callback(
    log_token_logprobs: bool,
    on_chunk: Callable[[ChatCompletionChunk, ChatCompletion], None] | None,
    stats: TaskResultStats,
) -> Callable[[ChatCompletionChunk, ChatCompletion], None] | None:
    "Create a callback function that logs token logprobs and/or calls the user's on_chunk callback if provided."
    if not log_token_logprobs and not on_chunk:
        return None

    def _on_chunk(chunk: ChatCompletionChunk, completion: ChatCompletion) -> None:
        if log_token_logprobs:
            stats.update(id=chunk.id, chunk=chunk)
        if on_chunk:
            on_chunk(chunk, completion)

    return _on_chunk


def _get_log_results_flags(log_results: bool | float | int, n: int) -> list[bool]:
    "Return a list of flags indicating whether to log results for each task."
    if isinstance(log_results, int) and log_results >= 1:
        result = [True] * log_results + [False] * max(n - log_results, 0)
    elif isinstance(log_results, float):
        result = [True] * int(log_results * n) + [False] * (n - int(log_results * n))
    else:
        result = [bool(log_results)] * n
    random.shuffle(result)
    return result


async def _get_task_result(
    task: Task,
    client: AsyncOpenAI,
    model: str,
    log_results: bool,
    n: int,
    cache: bool,
    log_dir: str | None,
    on_chunk: Callable[[ChatCompletionChunk, ChatCompletion], None] | None,
    semaphore: asyncio.Semaphore | None,
    token_scheduler: TokenScheduler | None,
    params: ChatCompletionParams | None,
    stats: TaskResultStats,
    transform: Callable[[TaskResult], T | Awaitable[T]],
) -> T:
    """
    Processes a single task by generating chat completions, grading responses, and calculating rewards.

    This is a helper function called by get_task_results for each task in the list. It:
    1. Makes n API calls to generate completions for the task
    2. Collects all completions and passes them to the task's grader function
    3. Tracks usage statistics and handles exceptions
    4. Calculates GRPO advantages based on reward distribution
    5. Applies the transform function to the TaskResult before returning

    See get_task_results for parameter descriptions.
    """
    # always request logprobs, unless explicitly disabled
    _params = (params or {}).copy()
    if "logprobs" not in _params:
        _params["logprobs"] = True
    elif _params["logprobs"] is None:
        del _params["logprobs"]
    chat_completions: list[ChatCompletion] = []
    rewards: dict[tuple[str, int], float] = {}
    metrics: dict[tuple[str, int], dict[str, float]] = {}
    exceptions: list[Exception] = []
    for chat_completion_future in asyncio.as_completed(
        get_chat_completion(
            client,
            cache=cache,
            log_dir=log_dir,
            log_results=log_results and i == 0,
            on_chunk=on_chunk,
            semaphore=semaphore,
            token_scheduler=token_scheduler,
            messages=task.messages,
            model=model,
            **_params,  # type: ignore
        )
        for i in range(n)
    ):
        try:
            chat_completion = await chat_completion_future
            chat_completions.append(chat_completion)
            stats.update(id=chat_completion.id, usage=chat_completion.usage)

            async def _grade(choice: Choice, grader: Grader) -> tuple[int, Grade]:
                return choice.index, await maybe_await(grader(choice))

            for grade_future in asyncio.as_completed(
                _grade(choice, task.grader) for choice in chat_completion.choices
            ):
                try:
                    choice_index, grade = await grade_future
                    reward, _metrics = (
                        grade if isinstance(grade, tuple) else (grade, {})
                    )
                    stats.update(id=chat_completion.id, reward=reward, metrics=_metrics)
                    rewards[chat_completion.id, choice_index] = reward
                    metrics[chat_completion.id, choice_index] = _metrics
                except Exception as e:
                    exceptions.append(e)
                    stats.update(id=chat_completion.id, exception=e)
                    continue
        except Exception as e:
            exceptions.append(e)
            stats.update(id=None, exception=e)
            continue
    if rewards:
        reward_mean = np.mean(list(rewards.values()))
        reward_std = np.std(list(rewards.values()))
        # calculate GRPO advantages
        advantages = {
            key: float((reward - reward_mean) / (reward_std + 1e-6))
            for key, reward in rewards.items()
        }
    else:
        advantages = {key: 0.0 for key in rewards.keys()}
    return await maybe_await(
        transform(
            TaskResult(
                task=task,
                chat_completions=chat_completions,
                rewards=rewards,
                metrics=metrics,
                advantages=advantages,
                exceptions=exceptions,
            )
        )
    )
