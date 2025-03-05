import json
import math
from openai.types.chat.chat_completion import Choice
import re
from typing import Iterable, TypedDict

from .tasks import Task


class TemporalCluePuzzle(TypedDict):
    num_clues: int
    prompt: str
    solution: dict[str, str]


def get_temporal_clue_puzzles() -> list[TemporalCluePuzzle]:
    return json.load(open("./data/bradhilton/temporal-clue/puzzles.json"))


def get_temporal_clue_tasks(
    reward_power: float = 1.0, surprise_bonus: float = 0.0
) -> Iterable[Task]:
    for puzzle in get_temporal_clue_puzzles():

        def grader(
            choice: Choice, puzzle: TemporalCluePuzzle = puzzle
        ) -> float | tuple[float, dict[str, float]]:
            content = choice.message.content
            assert isinstance(content, str)
            num_correct = 0
            for key, value in puzzle["solution"].items():
                if matches := re.findall(rf"{key}\. ([A-Za-z \.:-]+)", content):
                    match = matches[-1]
                    if match.strip().lower() == value.lower():
                        num_correct += 1
            acc = num_correct / len(puzzle["solution"])
            metrics = dict(acc=acc)
            if getattr(choice, "early_stop", False):
                metrics["early_stop"] = 1
            if surprise_bonus and choice.logprobs and choice.logprobs.content:
                matches = list(
                    re.finditer(
                        rf"[{''.join(puzzle["solution"])}]\. ([A-Za-z \.:-]+)",
                        b"".join(
                            bytes(token_logprob.bytes or [])
                            for token_logprob in choice.logprobs.content
                        ).decode(),
                    )
                )
                if not matches:
                    return acc
                last_end_index = matches[-1].end()
                logprobs = []
                cumulative_length = 0
                for token_logprob in choice.logprobs.content:
                    token_text = bytes(token_logprob.bytes or []).decode()
                    cumulative_length += len(token_text)
                    if not math.isnan(token_logprob.logprob):
                        logprobs.append(token_logprob.logprob)
                    if cumulative_length >= last_end_index:
                        break
                if logprobs:
                    surprise = -sum(logprobs) / len(logprobs)
                    metrics["surprise"] = surprise
                    return (
                        (1 - surprise_bonus) * acc + acc * surprise_bonus * surprise
                    ), metrics
            return acc**reward_power, metrics

        yield Task(
            messages=[
                {
                    "role": "user",
                    "content": puzzle["prompt"],
                }
            ],
            grader=grader,
        )
