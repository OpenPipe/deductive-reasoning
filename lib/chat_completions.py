import asyncio
from datetime import datetime, timedelta
import hashlib
import json
import obstore
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage
import os
from typing import Callable, Unpack


from .stream import consume_chat_completion_stream
from .token_schedulers import TokenScheduler, UnlimitedTokenScheduler
from .types import ChatCompletionRequest, ClientParams, CreateParams
from .utils import timeout

MAX_INT = 2**31 - 1
data_dir = os.path.abspath(os.getenv("DATA_DIR", "./data"))
chat_completions_dir = os.path.join(data_dir, "chat-completions")
chat_completion_logs_dir = os.path.join(data_dir, "chat-completion-logs")

os.makedirs(chat_completions_dir, exist_ok=True)
os.makedirs(chat_completion_logs_dir, exist_ok=True)
stores = [
    obstore.store.from_url(
        url=url,
        client_options={"timeout": timedelta(minutes=10)},
    )
    for url in (
        url
        for url in (
            # "memory:///",
            f"file://{data_dir}",
            os.getenv("OBJECT_STORE_URL"),
        )
        if url
    )
]
unlimited_semaphore = asyncio.Semaphore(MAX_INT)


async def get_chat_completion(
    client: AsyncOpenAI,
    cache: bool = True,
    log_dir: str | None = None,
    log_results: bool = True,
    on_chunk: Callable[[ChatCompletionChunk, ChatCompletion], None] | None = None,
    semaphore: asyncio.Semaphore | None = None,
    token_scheduler: TokenScheduler | None = None,
    **create_params: Unpack[CreateParams],
) -> ChatCompletion:
    """
    Given a client and arguments to openai.chat.completions.create, this function will return a chat completion with some additional features:
    - Caching of results to object storage
    - Logging of results to a file
    - Streaming of results to the callback function
    - Support for capping concurrent requests with a semaphore
    - Advanced support for token scheduling and breaking up long completions

    Args:
        client (AsyncOpenAI): An AsyncOpenAI client
        cache (bool): Whether to cache the results of the chat completion
        log_dir (str | None): The directory to log the results of the chat completion
        log_results (bool): Whether to log the results of the chat completion
        on_chunk (Callable[[ChatCompletionChunk, ChatCompletion], None]): A callback function that will be called with each chunk of the chat completion
        semaphore (asyncio.Semaphore): A semaphore to limit the number of concurrent requests
        token_scheduler (TokenScheduler): A token scheduler to limit the number of tokens used in the completion

    Returns:
        ChatCompletion: A chat completion
    """
    # Create a request object for caching
    request = ChatCompletionRequest(
        client_params=ClientParams(
            base_url=str(client.base_url),
            organization=client.organization,
            project=client.project,
        ),
        create_params=create_params,
    )

    if cache and (cached_completion := await _get_cached_completion(request)):
        return cached_completion
    async with semaphore or unlimited_semaphore:
        chat_completion = await _get_chat_completion(
            client,
            create_params,
            token_scheduler or UnlimitedTokenScheduler(),
            on_chunk=_create_on_chunk_callback(
                create_params, log_dir, log_results, on_chunk
            ),
        )
    if cache:
        await _cache_completion_result(request, chat_completion)
    return chat_completion


async def _get_cached_completion(
    request: ChatCompletionRequest,
) -> ChatCompletion | None:
    """Attempt to retrieve a chat completion from cache.

    Args:
        request: The chat completion request to look up in cache

    Returns:
        A cached ChatCompletion if found and valid, otherwise None
    """
    cache_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()

    # Fetch from all available stores in parallel
    results = await asyncio.gather(
        *[
            obstore.get_async(store, f"chat-completions/{cache_key}.json")
            for store in stores
        ],
        return_exceptions=True,
    )

    # Look for a valid result
    if result := next((r for r in results if not isinstance(r, BaseException)), None):
        # Parse the completion from JSON
        json_data = bytes(await result.bytes_async())
        chat_completion = ChatCompletion.model_validate_json(json_data)

        # Ensure it's a valid completion
        if _is_valid_chat_completion(chat_completion):
            # Replicate to stores that don't have it yet
            for store, result in zip(stores, results):
                if isinstance(result, FileNotFoundError):
                    await obstore.put_async(
                        store,
                        f"chat-completions/{cache_key}.json",
                        json_data,
                    )
            return chat_completion

    return None


def _create_on_chunk_callback(
    create_params: CreateParams,
    log_dir: str | None,
    log_results: bool,
    on_chunk: Callable[[ChatCompletionChunk, ChatCompletion], None] | None = None,
) -> Callable[[ChatCompletionChunk, ChatCompletion], None] | None:
    """Create a callback function for handling streaming chunks.

    This function sets up logging and wraps the user's callback function
    if provided, or returns None if no logging or callbacks are needed.

    Args:
        create_params: Parameters for the completion (used for conversation history)
        log_dir: Optional custom directory for logging
        log_results: Whether to log results
        on_chunk: Optional user callback for streaming

    Returns:
        A callback function or None if no callback is needed
    """
    if not (log_results or on_chunk):
        return None

    # Set up logging
    os.makedirs(log_dir or chat_completion_logs_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir or chat_completion_logs_dir, f"{datetime.now().isoformat()}.log"
    )

    # Write conversation history to the log file
    if log_results:
        with open(log_file, "w") as f:
            f.write(
                "".join(
                    f"{message['role'].capitalize()}:\n{message.get('content', '')}\n\n"
                    for message in create_params["messages"]
                )
                + "Assistant:\n"
            )

    # Create a callback function that handles both user callbacks and logging
    def callback(chunk: ChatCompletionChunk, completion: ChatCompletion) -> None:
        # Call user's callback if provided
        if on_chunk:
            on_chunk(chunk, completion)

        # Log chunk content if enabled
        if log_results and chunk.choices:
            try:
                with timeout():
                    with open(log_file, "a") as f:
                        f.write(chunk.choices[0].delta.content or "")
            except TimeoutError:
                pass  # Skip writing this chunk if it times out

    return callback


async def _cache_completion_result(
    request: ChatCompletionRequest,
    chat_completion: ChatCompletion,
) -> None:
    """Cache a chat completion result to all stores.

    Args:
        request: The original request object
        chat_completion: The completion to cache
    """
    cache_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    json_data = chat_completion.model_dump_json().encode()

    # Cache to all configured object stores
    await asyncio.gather(
        *[
            obstore.put_async(
                store,
                f"chat-completions/{cache_key}.json",
                json_data,
            )
            for store in stores
        ]
    )

    # Also log the request for analysis
    with open("./data/chat-completion-requests.jsonl", "a") as f:
        f.write(json.dumps(request, sort_keys=True) + "\n")


async def _get_chat_completion(
    client: AsyncOpenAI,
    create_params: CreateParams,
    token_scheduler: TokenScheduler,
    on_chunk: Callable[[ChatCompletionChunk, ChatCompletion], None] | None = None,
) -> ChatCompletion:
    """Internal function to handle chat completion with token scheduling.

    This function handles the actual API call to the chat completions endpoint,
    applying token scheduling constraints and supporting continuation of responses
    when they exceed token limits.

    Args:
        client: An AsyncOpenAI client or compatible API client
        create_params: Parameters for the completion creation request
        token_scheduler: Scheduler that manages token limits and continuation logic
        on_chunk: Optional callback function for streaming results

    Returns:
        A ChatCompletion object, potentially merged from multiple API calls
        if the response exceeded token limits
    """
    create_params = create_params.copy()
    async with token_scheduler.tokens(create_params) as max_completion_tokens:
        _create_params = create_params.copy()
        if max_completion_tokens:
            _create_params["max_completion_tokens"] = _create_params["max_tokens"] = (
                min(
                    _create_params.get(
                        "max_completion_tokens",
                        _create_params.get("max_tokens", MAX_INT),
                    )
                    or MAX_INT,
                    max_completion_tokens,
                )
            )
        if on_chunk:
            stream = await client.chat.completions.create(**_create_params, stream=True)
            chat_completion = await consume_chat_completion_stream(
                stream,
                on_chunk=on_chunk,
            )
        else:
            chat_completion = await client.chat.completions.create(**_create_params)
    # Check if we're done or need to continue generating
    if token_scheduler.is_finished(
        chat_completion, create_params, max_completion_tokens
    ):
        return chat_completion

    # Need to continue generating to complete the response
    # Set up for continuation by modifying the messages
    return await _continue_chat_completion(
        client, create_params, chat_completion, token_scheduler, on_chunk
    )


async def _continue_chat_completion(
    client: AsyncOpenAI,
    create_params: CreateParams,
    chat_completion: ChatCompletion,
    token_scheduler: TokenScheduler,
    _on_chunk: Callable[[ChatCompletionChunk, ChatCompletion], None] | None = None,
) -> ChatCompletion:
    """Continue a chat completion that was truncated due to token limits.

    There are two strategies for continuation:
    1. If continue_final_message=True: Update the last assistant message (assumes last message is from assistant)
    2. Otherwise: Add the current completion as an assistant message and continue generating

    Args:
        client: The OpenAI client
        create_params: Parameters for the API request
        chat_completion: The current partial completion
        token_scheduler: The token scheduler managing the continuation
        _on_chunk: Optional callback for streaming chunks

    Returns:
        A new completion with continued content
    """
    # Make a copy of the messages to avoid modifying the original
    messages = create_params["messages"] = list(create_params["messages"])

    # We currently only support continuing the first choice
    # TODO: Support multiple choices
    assistant_content = chat_completion.choices[0].message.content
    assert assistant_content is not None

    # Prepare the extra_body parameters for continuation
    extra_body = create_params.setdefault("extra_body", {})
    assert isinstance(extra_body, dict)

    if extra_body.get("continue_final_message"):
        # Strategy 1: Continue the last assistant message
        assert messages[-1]["role"] == "assistant"
        messages[-1] = messages[-1].copy()
        messages[-1]["content"] = assistant_content
    else:
        # Strategy 2: Add the current completion as an assistant message and continue
        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
            }
        )
        extra_body["add_generation_prompt"] = False
        extra_body["continue_final_message"] = True

    # Make the next API call to continue the completion
    new_completion = await _get_chat_completion(
        client,
        create_params,
        token_scheduler,
        _on_chunk,
    )

    # Merge the original completion with the continuation
    _merge_chat_completions(chat_completion, new_completion)
    return new_completion


def _merge_chat_completions(
    original: ChatCompletion,
    into: ChatCompletion,
) -> None:
    for original_choice, into_choice in zip(original.choices, into.choices):
        _merge_choices(original_choice, into_choice)
    if original.usage and into.usage:
        _merge_usage(original.usage, into.usage)


def _merge_choices(original: Choice, into: Choice) -> None:
    if original.logprobs and into.logprobs:
        _merge_choice_logprobs(original.logprobs, into.logprobs)
    if original.message.content and into.message.content:
        into.message.content = original.message.content + into.message.content


def _merge_choice_logprobs(original: ChoiceLogprobs, into: ChoiceLogprobs) -> None:
    if original.content and into.content:
        into.content = original.content + into.content
    if original.refusal and into.refusal:
        into.refusal = original.refusal + into.refusal


def _merge_usage(original: CompletionUsage, into: CompletionUsage) -> None:
    into.prompt_tokens = original.prompt_tokens
    into.completion_tokens += original.completion_tokens
    into.total_tokens = into.prompt_tokens + into.completion_tokens


def _is_valid_chat_completion(chat_completion: ChatCompletion) -> bool:
    return bool(
        chat_completion.choices
        and all(
            choice.message.content or choice.message.refusal
            for choice in chat_completion.choices
        )
    )
