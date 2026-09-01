# What: the loop target is boxed to be passed to the awaited call, and the box
# has to be released on the loop's EXIT edge as well as its back edge -- only
# running it shows the values arrive, and the leak gate is what shows the
# reference did not survive the loop.
import asyncio


async def double(n: int) -> int:
    return n * 2


async def label(n: int) -> str:
    return "v" + str(n)


async def main() -> None:
    out = []
    for i in range(4):
        out.append(await double(i))
    print(out)

    tags = []
    for i in range(3):
        tags.append(await label(i))
    print(tags)

    total = 0
    for i in range(5):
        total += await double(i)
    print(total)

    j = 0
    while j < 3:
        print(await double(j))
        j += 1

    nested = []
    for i in range(2):
        for k in range(2):
            nested.append(await double(i + k))
    print(nested)

    print([await double(i) for i in range(3)])


asyncio.run(main())
