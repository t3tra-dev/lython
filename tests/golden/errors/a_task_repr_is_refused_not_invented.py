# WHAT: repr() of an asyncio Task raises NotImplementedError naming why,
# rather than printing text CPython would not: CPython's names the task's
# coroutine with the file and line it runs at, which this runtime does not
# keep.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the refusal is raised by
# the running program, from the task the program made.
import asyncio


async def work() -> int:
    await asyncio.sleep(0)
    return 5


async def main() -> None:
    t = asyncio.create_task(work())
    print(t)


asyncio.run(main())
