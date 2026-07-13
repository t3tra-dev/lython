import asyncio


async def add(a: int, b: int) -> int:
    return a + b


async def main() -> None:
    print(await add(1, 2))


asyncio.run(main())
