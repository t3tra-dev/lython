# What: `async for` over a class whose `__anext__` is a coroutine that awaits
# before it returns. The loop suspends inside `__anext__`, so another task runs
# between its items, and it ends on StopAsyncIteration. The interleaving is the
# evidence of the suspension, and only running it shows the order.
import asyncio


class Ticker:
    def __init__(self, n: int) -> None:
        self.i = 0
        self.n = n

    def __aiter__(self) -> "Ticker":
        return self

    async def __anext__(self) -> int:
        if self.i >= self.n:
            raise StopAsyncIteration
        await asyncio.sleep(0)
        self.i += 1
        return self.i * 10


async def chatter(n: int) -> None:
    for i in range(n):
        print("chatter", i)
        await asyncio.sleep(0)


async def main() -> int:
    other = asyncio.create_task(chatter(3))
    total = 0
    async for v in Ticker(3):
        print("got", v)
        total += v
    await other
    return total


print(asyncio.run(main()))
