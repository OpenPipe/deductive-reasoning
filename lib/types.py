from openai._types import Body, Headers, Query
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from typing import Never, TypedDict


class CreateParams(CompletionCreateParamsBase, total=False):
    """Parameters for chat completion creation with additional fields."""
    extra_headers: Headers
    extra_query: Query
    extra_body: Body


class ClientParams(TypedDict):
    """Parameters describing an OpenAI client."""
    base_url: str
    organization: str | None
    project: str | None


class ChatCompletionRequest(TypedDict):
    """A request to the chat completion API."""
    client_params: ClientParams
    create_params: CreateParams


class ChatCompletionParams(CreateParams, total=False):
    """Parameters for chat completion with restricted fields."""
    messages: Never  # type: ignore
    model: Never  # type: ignore