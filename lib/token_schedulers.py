from contextlib import asynccontextmanager
from openai.types.chat.chat_completion import ChatCompletion
from typing import AsyncContextManager, AsyncGenerator, Protocol

from .types import CreateParams


class TokenScheduler(Protocol):
    """Protocol for token scheduling in chat completion API calls.
    
    Token schedulers help manage token limits when generating completions,
    especially for handling long responses by breaking them into multiple
    API calls as needed.
    """
    
    def tokens(self, params: CreateParams) -> AsyncContextManager[int]:
        """Get the token limit for the current API call.
        
        Args:
            params: The parameters for the chat completion API call
            
        Returns:
            An async context manager that yields the maximum number of tokens
            to use for the current API call (0 means no limit)
        """
        ...

    def is_finished(
        self,
        chat_completion: ChatCompletion,
        params: CreateParams,
        max_completion_tokens: int,
    ) -> bool:
        """Determine if the response generation is complete.
        
        Args:
            chat_completion: The current chat completion result
            params: The parameters used for the chat completion API call
            max_completion_tokens: The maximum token limit that was applied
            
        Returns:
            True if the response is complete, False if it needs to continue
            with another API call
        """
        ...


class UnlimitedTokenScheduler(TokenScheduler):
    """A simple token scheduler with no limits or continuation logic.
    
    This scheduler doesn't impose any token limits and always considers
    responses complete after a single API call.
    """
    
    def tokens(self, _: CreateParams) -> AsyncContextManager[int]:
        """Return an unlimited token context.
        
        Args:
            _: Ignored parameters
            
        Returns:
            An async context manager yielding 0 (unlimited tokens)
        """
        @asynccontextmanager
        async def unlimited_tokens() -> AsyncGenerator[int, None]:
            try:
                yield 0
            finally:
                pass

        return unlimited_tokens()

    def is_finished(
        self,
        _: ChatCompletion,
        __: CreateParams,
        ___: int,
    ) -> bool:
        """Always consider the response complete.
        
        Args:
            _: Ignored chat completion
            __: Ignored parameters
            ___: Ignored token limit
            
        Returns:
            Always True, indicating no need for continuation
        """
        return True