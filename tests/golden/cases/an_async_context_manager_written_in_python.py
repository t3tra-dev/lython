# What: `async with` over a class whose `__aenter__` / `__aexit__` are Python
# methods. Running it is what shows the order the two ran in, that the value
# `as` bound is the one `__aenter__` returned, and that a True from `__aexit__`
# suppresses -- none of which a compile-time check can see.
import asyncio


class Span:
    def __init__(self, name: str) -> None:
        self.name = name

    async def __aenter__(self) -> "Span":
        print("enter", self.name)
        return self

    async def __aexit__(self, kind: object, value: object, tb: object) -> bool:
        print("exit", self.name)
        return False


class Swallow:
    async def __aenter__(self) -> "Swallow":
        return self

    async def __aexit__(self, kind: object, value: object, tb: object) -> bool:
        print("swallowed")
        return True


async def main() -> None:
    async with Span("outer") as span:
        print("body", span.name)
    try:
        async with Span("raising"):
            raise ValueError("boom")
    except ValueError as error:
        print("caught", error)
    async with Swallow():
        raise KeyError("gone")
    print("still running")


asyncio.run(main())
