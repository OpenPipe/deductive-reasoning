from dataclasses import dataclass, field
import math
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


@dataclass
class InferenceEarlyStop:
    """
    Utility for stopping inference early if token log probabilities are too low.

    Args:
        alpha: The smoothing factor for the exponential weighted moving average.
        threshold: The log probability threshold to stop inference below.
        log_early_stops: Whether to log early stops.
        log_last_n_characters: The number of characters to log from the end of the stopped completion.
    """

    alpha: float = 0.992
    threshold: float = -3
    log_early_stops: bool = False
    log_last_n_characters: int = 64
    ewm_logprobs: dict[str, float] = field(default_factory=dict)

    def __call__(self, chunk: ChatCompletionChunk, completion: ChatCompletion) -> None:
        # TODO: handle multiple choices and refusal logprobs
        if (
            not chunk.choices
            or not chunk.choices[0].logprobs
            or not chunk.choices[0].logprobs.content
        ):
            return
        for token_logprob in chunk.choices[0].logprobs.content:
            if token_logprob.logprob is None or math.isnan(token_logprob.logprob):
                raise StopIteration()
            ewm_logprob = (
                self.alpha * self.ewm_logprobs.get(completion.id, 0)
                + (1 - self.alpha) * token_logprob.logprob
            )
            if ewm_logprob < self.threshold:
                if self.log_early_stops:
                    print(
                        f"Early stopping - ewm_logprob: {ewm_logprob} completion_tokens: {len(completion.choices[0].logprobs.content)}"  # type: ignore
                    )
                    if self.log_last_n_characters:
                        print(
                            f"Last {self.log_last_n_characters} characters: {completion.choices[0].message.content[-self.log_last_n_characters :]}"  # type: ignore
                        )
                setattr(completion.choices[0], "early_stop", True)
                raise StopIteration()
            self.ewm_logprobs[completion.id] = ewm_logprob
