# What: a callback the host calls after the program's main body has returned
# still runs -- the runtime stays up while the host holds a callback -- with
# the state its closure captured, and the host's global functions (here
# queueMicrotask, typed as Window's method) are called like any other.
#
# Why run: the call happens after main, from the host's event loop.
import js
from js import queueMicrotask

counter: list[int] = [0]


def make(label: str) -> None:
    def tick() -> None:
        counter[0] += 1
        print(label, counter[0])

    queueMicrotask(tick)
    js.queueMicrotask(tick)


make("tick")
print("main done")
