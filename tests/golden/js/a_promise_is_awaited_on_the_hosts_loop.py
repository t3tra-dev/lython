# What: `await` on a JavaScript Promise inside tasks the host's event loop
# runs after the main body returns: a resolved value arrives typed by the
# Promise's argument, `gather` interleaves Promise waits with timers, a
# rejection raises RuntimeError with the host's message, a value of the wrong
# type is caught where it arrives instead of stalling the task, and a task
# that calls `asyncio.run()` is told the loop is running.
#
# Why run: every line after "scheduled" happens in callbacks the host makes,
# in an order only the run shows.
import asyncio
from js import Promise


async def fetch_number(n: int) -> int:
    v = await Promise.resolve(n)
    await asyncio.sleep(0.005 * n)
    return v * 10


def explode(v: int) -> int:
    raise ValueError("boom " + str(v))


async def main() -> None:
    print("text", await Promise.resolve("hi"))
    try:
        await Promise.resolve(1).then(explode)
    except RuntimeError as e:
        print("rejected:", e)
    print(await asyncio.gather(fetch_number(3), fetch_number(1), fetch_number(2)))
    wrong: Promise[int] = Promise.resolve("not a number")
    try:
        await wrong
    except RuntimeError as e:
        print("mismatch:", e)
    try:
        asyncio.run(fetch_number(1))
    except RuntimeError as e:
        print("run:", e)


asyncio.create_task(main())
print("scheduled")
