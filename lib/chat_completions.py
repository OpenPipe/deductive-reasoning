import asyncio
from datetime import datetime
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
import os
from typing import Callable, Unpack


from .stream import consume_chat_completion_stream
from .types import CreateParams
from .utils import timeout

MAX_INT = 2**31 - 1
unlimited_semaphore = asyncio.Semaphore(MAX_INT)


async def get_chat_completion(
    client: AsyncOpenAI,
    log_dir: str | None = None,
    log_results: bool = True,
    on_chunk: Callable[[ChatCompletionChunk, ChatCompletion], None] | None = None,
    semaphore: asyncio.Semaphore | None = None,
    **create_params: Unpack[CreateParams],
) -> ChatCompletion:
    """
    Given a client and arguments to openai.chat.completions.create, this function will return a chat completion with some additional features:
    - Logging of results to a file
    - Streaming of results to the callback function
    - Support for capping concurrent requests with a semaphore

    Args:
        client (AsyncOpenAI): An AsyncOpenAI client
        log_dir (str | None): The directory to log the results of the chat completion
        log_results (bool): Whether to log the results of the chat completion
        on_chunk (Callable[[ChatCompletionChunk, ChatCompletion], None]): A callback function that will be called with each chunk of the chat completion
        semaphore (asyncio.Semaphore): A semaphore to limit the number of concurrent requests

    Returns:
        ChatCompletion: A chat completion
    """
    async with semaphore or unlimited_semaphore:
        on_chunk = _create_on_chunk_callback(
            create_params, log_dir, log_results, on_chunk
        )
        if on_chunk:
            return await consume_chat_completion_stream(
                await client.chat.completions.create(**create_params, stream=True),
                on_chunk=on_chunk,
            )
        else:
            return await client.chat.completions.create(**create_params)


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
    log_dir = log_dir or "./logs/chat-completions"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().isoformat()}.log")

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
