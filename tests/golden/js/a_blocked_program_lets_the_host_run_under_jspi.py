# What: a program that blocks -- `asyncio.run()` awaiting Promises, and
# `time.sleep` -- lets the host's event loop run while it waits, under JSPI
# (the WASI loader, .jspi.stdout): the host timer fires during the sleep and
# the gathered promises settle inside `run`. Emscripten cannot suspend the
# program, so there the timer fires only after the main body returns and
# `run` raises at the first Promise.
#
# Why run: what fires when, and whether the program could wait at all, is
# the host's run-time behaviour.
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
