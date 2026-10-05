# What: a coroutine's local, bound before a `try` whose body is an
# `async with` and read after the `try`, is still alive when it is read, and
# is released once -- on the path through the body, through the handler, and
# through a `finally`.
# WHY THIS IS RUN: the defect was in where the release went, which compiled
# (once it compiled at all) and freed the object before the read; only
# reading it shows the read is of a live object.
import asyncio


class Span:
    def __init__(self, name: str) -> None:
        self.name = name

    async def __aenter__(self) -> "Span":
        return self

    async def __aexit__(self, kind: object, value: object, tb: object) -> bool:
        return False


async def handled(fail: bool) -> None:
    span = Span("handled")
    try:
        async with Span("inner"):
            if fail:
                raise ValueError("boom")
            print("body")
    except ValueError:
        print("caught")
    print(span.name)


async def finished() -> None:
    span = Span("finished")
    try:
        async with Span("inner"):
            print("body")
    finally:
        print("finally")
    print(span.name)


asyncio.run(handled(False))
asyncio.run(handled(True))
asyncio.run(finished())
