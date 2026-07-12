import asyncio


async def zero() -> int:
    print("before")
    await asyncio.sleep(0.0)
    print("after")
    return 7


async def timed() -> int:
    print("start")
    await asyncio.sleep(0.01)
    print("woke")
    return 3


async def run_all() -> int:
    a = await zero()
    b = await timed()
    return a + b


print(asyncio.run(run_all()))
