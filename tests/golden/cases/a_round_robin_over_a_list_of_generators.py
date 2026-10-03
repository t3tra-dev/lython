# A scheduler's shape: generators taken off a list, resumed, and put back
# until each is exhausted. The list is built from the generator function, so
# each element keeps its frame; the interleaving is the run-time order the
# resumes produce.
from typing import Generator


def worker(name: str, n: int) -> Generator[int, None, None]:
    for i in range(n):
        print(name, i)
        yield i


def run() -> None:
    tasks = [worker("a", 2), worker("b", 3)]
    while tasks:
        task = tasks.pop(0)
        try:
            next(task)
            tasks.append(task)
        except StopIteration:
            print("done")


run()
