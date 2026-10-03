# What: `asyncio.run()` waits on the host by suspending the program (JSPI),
# which a callback the host is running cannot do -- that is the host's own
# stack. Run from the program, a coroutine awaiting a Promise gets its result;
# run from a timer callback, the same call raises RuntimeError instead of
# spinning forever, and a coroutine that waits only on Python still finishes.
#
# Why run: whether the program can be suspended is the run-time state of the
# host's stack.
import asyncio
from js import Promise, setTimeout


async def pure() -> int:
    await asyncio.sleep(0.01)
    return 7


async def hosted() -> int:
    return await Promise.resolve(8)


def from_callback() -> None:
    print("callback pure", asyncio.run(pure()))
    try:
        print("callback hosted", asyncio.run(hosted()))
    except RuntimeError as e:
        print("callback hosted:", e)


print("hosted", asyncio.run(hosted()))
setTimeout(from_callback, 0)
print("main body done")
