from contextlib import contextmanager
import signal
from typing import Generator


@contextmanager
def timeout(seconds: int = 1) -> Generator[None, None, None]:
    def timeout_handler(signum: object, frame: object) -> None:
        raise TimeoutError()

    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    try:
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)
