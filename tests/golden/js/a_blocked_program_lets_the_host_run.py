# What: `asyncio.run()` blocks the program and still lets the host's event
# loop run, because the loader suspends it (JSPI) whenever the loop waits:
# the gathered promises settle inside `run`, and a host timer set before it
# fires during its waits. `time.sleep` does not suspend -- it blocks the host
# like a sleep on its thread -- so the timer has not fired by "slept".
#
# Why run: what fires when is the host's run-time behaviour.
import asyncio
import time
from js import Promise, setTimeout


async def slow(n: int) -> int:
    v = await Promise.resolve(n)
    await asyncio.sleep(0.01 * n)
    return v * 10


async def main() -> int:
    print("first", await Promise.resolve(1))
    results = await asyncio.gather(slow(3), slow(1), slow(2))
    print(results)
    return sum(results)


def tick() -> None:
    print("host timer")


setTimeout(tick, 5)
time.sleep(0.05)
print("slept")
try:
    print("total", asyncio.run(main()))
except RuntimeError as e:
    print("run:", e)
print("main body done")
