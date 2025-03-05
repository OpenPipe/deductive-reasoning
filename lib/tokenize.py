from dataclasses import dataclass
from itertools import takewhile
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
import random
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import cast


from .tasks import TaskResult


@dataclass
class TokenizedResult:
    conversation: list
    reward: float
    advantage: float
    deferred: bool
    chat_template: str
    chat: str
    tokens: list[str]
    token_ids: list[int]
    input_pos: list[int]
    assistant_mask: list[int]
    token_logprobs: list[ChatCompletionTokenLogprob] | None
    prompt_id: int = 0
    prompt_length: int = 0

    def without_prompt(self) -> "TokenizedResult":
        assistant_mask = self.assistant_mask[self.prompt_length :]
        return TokenizedResult(
            conversation=self.conversation,
            advantage=self.advantage,
            reward=self.reward,
            deferred=self.deferred,
            chat_template=self.chat_template,
            chat=self.chat,
            tokens=self.tokens[self.prompt_length :],
            token_ids=self.token_ids[self.prompt_length :],
            input_pos=self.input_pos[self.prompt_length :],
            assistant_mask=assistant_mask,
            token_logprobs=(
                self.token_logprobs[len(self.token_logprobs) - sum(assistant_mask) :]
                if self.token_logprobs is not None
                else None
            ),
            prompt_id=self.prompt_id,
            prompt_length=0,
        )


class TaskResultTokenizer:
    def __init__(
        self,
        pretrained_tokenizer_or_model_name_or_path: (
            PreTrainedTokenizer | PreTrainedTokenizerFast | str
        ),
    ) -> None:
        self.tokenizer = (
            AutoTokenizer.from_pretrained(pretrained_tokenizer_or_model_name_or_path)
            if isinstance(pretrained_tokenizer_or_model_name_or_path, str)
            else pretrained_tokenizer_or_model_name_or_path
        )

    def __call__(self, task_result: TaskResult) -> list[TokenizedResult]:
        chat_completions = task_result.chat_completions.copy()
        random.shuffle(chat_completions)
        tokenized_results = [
            self._tokenized_result(
                task_result,
                choice,
                task_result.rewards.get((chat_completion.id, choice.index), 0),
                task_result.advantages.get((chat_completion.id, choice.index), 0),
            )
            for chat_completion in chat_completions
            for choice in chat_completion.choices
        ]
        prompt_id = random.randint(-(2**63), 2**63 - 1)
        prompt_length = len(
            list(
                takewhile(
                    lambda x: len(set(x)) == 1,
                    zip(*(r.token_ids for r in tokenized_results)),
                )
            )
        )
        for result in tokenized_results:
            result.prompt_id = prompt_id
            result.prompt_length = prompt_length
            # zero out assistant prompt tokens
            result.assistant_mask[:prompt_length] = [0] * prompt_length
        return tokenized_results

    def _tokenized_result(
        self, task_result: TaskResult, choice: Choice, reward: float, advantage: float
    ) -> TokenizedResult:
        conversation: list = task_result.task.messages + [
            {
                "role": "assistant",
                "content": choice.message.content,
            }
        ]
        chat_template = update_chat_template(self.tokenizer.get_chat_template())
        chat = cast(
            str,
            self.tokenizer.apply_chat_template(
                conversation, chat_template=chat_template, tokenize=False
            ),
        )
        tokenized_result = cast(
            dict[str, list[int]],
            self.tokenizer.apply_chat_template(
                conversation,
                chat_template=chat_template,
                return_dict=True,
                return_assistant_tokens_mask=True,
            ),
        )
        if (
            choice.logprobs
            and choice.logprobs.content
            and choice.logprobs.content[0].token.startswith("token_id:")
        ):
            start = tokenized_result["assistant_masks"].index(1)
            try:
                end = start + tokenized_result["assistant_masks"][start:].index(0)
            except ValueError:
                end = len(tokenized_result["assistant_masks"])
            tokenized_result["input_ids"][start:end] = [
                int(token_logprob.token.split(":")[1])
                for token_logprob in choice.logprobs.content
            ]
            tokenized_result["assistant_masks"][start:end] = [
                1 for _ in choice.logprobs.content
            ]
            token_logprobs = choice.logprobs.content
        else:
            token_logprobs = None
        tokens = [
            self.tokenizer.decode(token_id)
            for token_id in tokenized_result["input_ids"]
        ]
        if token_logprobs is None:
            token_logprobs = self.get_token_logprobs(
                choice,
                [
                    token
                    for token, mask in zip(tokens, tokenized_result["assistant_masks"])
                    if mask
                ],
            )
        return TokenizedResult(
            conversation=conversation,
            reward=reward,
            advantage=advantage,
            deferred=False,
            chat_template=chat_template,
            chat=chat,
            tokens=tokens,
            token_ids=tokenized_result["input_ids"],
            input_pos=list(range(len(tokens))),
            assistant_mask=tokenized_result["assistant_masks"],
            token_logprobs=token_logprobs,
        )

    def get_token_logprobs(
        self,
        choice: Choice,
        assistant_tokens: list[str],
    ) -> list[ChatCompletionTokenLogprob] | None:
        if not choice.logprobs:
            return None
        if not choice.logprobs.content:
            return None
        result_token_logprobs = choice.logprobs.content.copy()
        if "".join(assistant_tokens) != "".join(
            token_logprob.token for token_logprob in result_token_logprobs
        ) and len(assistant_tokens) != len(result_token_logprobs):
            print("Assistant tokens are not equal, skipping token logprobs")
            return None
        elif assistant_tokens == [
            token_logprob.token for token_logprob in result_token_logprobs
        ]:
            return result_token_logprobs
        else:
            completion = ""
            result_completion = ""
            token_logprobs = []
            try:
                while True:
                    if completion == result_completion:
                        token = assistant_tokens.pop(0)
                        result_token_logprob = result_token_logprobs.pop(0)
                        result_token = result_token_logprob.token
                        if token == result_token:
                            token_logprobs.append(result_token_logprob)
                        else:
                            token_logprobs.append(
                                ChatCompletionTokenLogprob(
                                    token=token,
                                    logprob=float("nan"),
                                    top_logprobs=[],
                                )
                            )
                        completion += token
                        result_completion += result_token
                    elif len(completion) < len(result_completion):
                        token = assistant_tokens.pop(0)
                        token_logprobs.append(
                            ChatCompletionTokenLogprob(
                                token=token,
                                logprob=float("nan"),
                                top_logprobs=[],
                            )
                        )
                        completion += token
                    elif len(completion) > len(result_completion):
                        result_completion += result_token_logprobs.pop(0).token
                    else:
                        print("Warning: Completions are not equal")
                        print(f"Completion: {completion}")
                        print(f"Result completion: {result_completion}")
                        token_logprobs = None
                        break
            except IndexError:
                pass
        return token_logprobs


def update_chat_template(chat_template: str) -> str:
    return (
        chat_template
        # Remove template logic that strips reasoning content from the chat messages
        .replace(
            "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}",
            "",
        )
        # Add generation tags for assistant token masking
        .replace(
            "{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}",
            "{{'<｜Assistant｜>'}}{% generation %}{{ content }}{% endgeneration %}{{'<｜end▁of▁sentence｜>'}}",
        )
        # Add generation tags for assistant token masking (for Hermes 2 Theta)
        .replace(
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}",
            "{{'<|im_start|>' + message['role'] + '\n'}}{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] }}{% endgeneration %}{% else %}{{ message['content'] }}{% endif %}{{'<|im_end|>' + '\n'}}",
        )
        # Add generation tags for assistant token masking (for Qwen 2.5 Instruct)
        .replace(
            """
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    """.strip(),
            """
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" and not message.tool_calls %}
        {{- '<|im_start|>' + message.role + '\\n' }}{% generation %}{{ message.content }}{% endgeneration %}{{ '<|im_end|>' + '\\n' }}
""".strip(),
        ).replace(
            """
        {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}""".strip(),
            """
        {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' }}{% generation %}{{ message.content }}{% endgeneration %}
        {%- endif %}""".strip(),
        )
        # Add generation tags for assistant token masking (for Llama 3.3 70B)
        .replace(
            "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}",
            "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}{%- if message['role'] == 'assistant' %}{% generation %}{{ message['content'] | trim + '<|eot_id|>' }}{% endgeneration %}{% else %}{{ message['content'] | trim + '<|eot_id|>' }}{% endif %}",
        )
    )
