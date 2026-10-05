# What: a Future whose exception nobody asked for reports it on stderr when it
# dies, as CPython's asyncio does: the message, the future's repr, the
# exception. One whose exception() or result() was called reports nothing.
# WHY THIS IS RUN: the report is written by the future's finalizer, so when --
# and whether -- it appears is only seen by running the program.
import asyncio


async def main() -> None:
    lost = asyncio.Future[int]()
    lost.set_exception(ValueError("bad"))
    seen = asyncio.Future[int]()
    seen.set_exception(KeyError("k"))
    print(seen.exception())
    read = asyncio.Future[int]()
    read.set_exception(KeyError("r"))
    try:
        read.result()
    except KeyError:
        print("result raised")
    print("end main")


asyncio.run(main())
print("after run")
