import asyncio
import black
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import signal
from typing import Generator


def black_print(object: object) -> None:
    print(black.format_str(str(object), mode=black.Mode()))


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


def symlink_shm(relative_path: str) -> str | None:
    if not Path("/dev/shm").exists():
        return None
    output_dir = Path(relative_path).absolute()
    # Create corresponding path in /dev/shm by using the relative parts
    shm_dir = Path("/dev/shm").joinpath(*Path(relative_path).parts)
    if output_dir.is_symlink():
        return shm_dir.as_posix()
    os.makedirs(output_dir.parent, exist_ok=True)
    if output_dir.exists():
        # copy output_dir to shm_dir
        shutil.copytree(output_dir, shm_dir)
        # delete output_dir
        shutil.rmtree(output_dir)
    else:
        os.makedirs(shm_dir, exist_ok=True)
    os.symlink(shm_dir, output_dir, target_is_directory=True)
    return shm_dir.as_posix()


async def rsync_dir(
    relative_path: str, destination: str, destination_subpath: str = ""
) -> None:
    abs_path = Path(relative_path).absolute().as_posix()
    if destination_subpath:
        destination = Path(destination).joinpath(destination_subpath).as_posix()
    else:
        destination = Path(destination).joinpath(*Path(relative_path).parts).as_posix()
    print(f"rsyncing {abs_path} to {destination}")
    os.makedirs("./logs", exist_ok=True)
    with open("./logs/rsync.log", "w") as log_file:
        process = await asyncio.create_subprocess_shell(
            f"gsutil -m rsync -r -d {abs_path} {destination.replace("gs:/", "gs://")}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def log_output(stream):
            while True:
                line = await stream.readline()
                if not line:
                    break
                log_file.write(line.decode())
                log_file.flush()

        await asyncio.gather(log_output(process.stdout), log_output(process.stderr))
        await process.wait()
