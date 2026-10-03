# What: `asyncio.run()` on a JavaScript host blocks: a coroutine that waits
# only on Python (here a timer) runs to its result. One that waits on a
# Promise needs the host to run meanwhile: under JSPI (the WASI loader,
# .jspi.stdout) the loop suspends until it has, and the result arrives; under
# Emscripten, which cannot suspend, it raises RuntimeError rather than
# spinning forever.
#
# Why run: the refusal is a run-time state of the loop, not a static fact.
import asyncio
from js import Promise


async def pure() -> int:
    await asyncio.sleep(0.01)
    return 7


async def hosted() -> int:
    return await Promise.resolve(8)


print("pure", asyncio.run(pure()))
try:
    print("hosted", asyncio.run(hosted()))
except RuntimeError as e:
    print("hosted:", e)
