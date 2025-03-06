import json
from openai.types.chat.chat_completion import Choice
import re
from typing import Iterable, TypedDict

from .tasks import Task


class TemporalCluePuzzle(TypedDict):
    num_clues: int
    prompt: str
    solution: dict[str, str]


def get_temporal_clue_puzzles() -> list[TemporalCluePuzzle]:
    return json.load(open("./data/puzzles.json"))


def get_temporal_clue_tasks() -> Iterable[Task]:
    for puzzle in get_temporal_clue_puzzles():

        def grader(choice: Choice, puzzle: TemporalCluePuzzle = puzzle) -> float:
            content = choice.message.content
            assert isinstance(content, str)
            num_correct = 0
            for key, value in puzzle["solution"].items():
                if matches := re.findall(rf"{key}\. ([A-Za-z \.:-]+)", content):
                    match = matches[-1]
                    if match.strip().lower() == value.lower():
                        num_correct += 1
            return num_correct / len(puzzle["solution"])

        yield Task(
            messages=[
                {
                    "role": "user",
                    "content": puzzle["prompt"],
                }
            ],
            grader=grader,
        )
