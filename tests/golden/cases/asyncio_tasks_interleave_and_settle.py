# What: asyncio written in Python, on coroutines the generator state machine
# suspends. Tasks interleave at each `sleep(0)`, `gather` keeps argument
# order, timers wake in deadline order rather than creation order, a task's
# exception reaches its awaiter, `cancel()` lands as CancelledError at the
# task's next suspension, and a Future resolved from a loop callback wakes its
# awaiter. All of it is the order things happen in at run time, which no
# compile-time check sees.
import asyncio


async def worker(name: str, steps: int, log: list[str]) -> int:
    for i in range(steps):
        log.append(name + str(i))
        await asyncio.sleep(0)
    return steps * 10


async def fails() -> int:
    await asyncio.sleep(0)
    raise ValueError("bad worker")


async def sleeper(delay: float, label: str, log: list[str]) -> str:
    await asyncio.sleep(delay)
    log.append(label)
    return label


async def forever() -> int:
    while True:
        await asyncio.sleep(0)


async def main() -> None:
    log: list[str] = []
    a = asyncio.create_task(worker("a", 3, log))
    b = asyncio.create_task(worker("b", 2, log))
    print(await a, await b)
    print(log)

    print(await asyncio.gather(worker("x", 1, log), worker("y", 2, log)))

    order: list[str] = []
    late = asyncio.create_task(sleeper(0.02, "late", order))
    early = asyncio.create_task(sleeper(0.01, "early", order))
    print(await late, await early, order)

    t = asyncio.create_task(fails())
    try:
        await t
    except ValueError as e:
        print("caught", e, t.done())

    spin = asyncio.create_task(forever())
    await asyncio.sleep(0)
    print("cancel", spin.cancel())
    try:
        await spin
    except asyncio.CancelledError:
        print("cancelled", spin.cancelled())

    fut = asyncio.Future[int]()
    asyncio.get_running_loop().call_soon(lambda: fut.set_result(42))
    print("future", await fut)


asyncio.run(main())
