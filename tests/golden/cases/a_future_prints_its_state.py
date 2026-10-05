# WHAT: an asyncio Future prints as CPython's does in each state it can be
# printed in -- pending, finished with a result, finished with an exception
# (inside a list too), cancelled.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the state is the
# running program's, and what was wrong was the text: the port had no
# __repr__, so each printed `<asyncio.Future object at 0x...>`.
import asyncio


async def main() -> None:
    f: asyncio.Future[int] = asyncio.Future()
    print(repr(f))
    f.set_result(3)
    print(f)
    g: asyncio.Future[str] = asyncio.Future()
    g.set_exception(ValueError("bad"))
    print([g])
    g.exception()
    h: asyncio.Future[float] = asyncio.Future()
    h.cancel()
    print(str(h))


asyncio.run(main())
