"""Asynchronous I/O: the event loop, futures and tasks.

A port of CPython's pure-Python asyncio (Lib/asyncio/events.py,
base_events.py, futures.py, tasks.py), restricted to what types statically.
A coroutine here is a generator body run by the state machine every
generator runs in (docs/async-design.md): `await x` is `yield from` over
what `x` awaits with, a future's `__await__` yields the future itself, and a
task steps its coroutine with `send(None)` and `throw()` and waits on what it
yielded -- the protocol CPython's Task speaks.

Deviations from CPython:
  - A callback takes no arguments: `call_soon(fn)` with `fn` a
    `Callable[[], None]`, not `call_soon(fn, *args)`. Bind the arguments with
    a lambda. `add_done_callback(fn)` passes the future, as CPython does.
  - `Task` is not a `Future` subclass: a generic class cannot yet derive
    from a generic base with its own type parameter (`class B(A[T])` makes
    `builtins.T` a contract). Both derive from
    `_Waiter`, which is what a task waits on; `isinstance(task, Future)` is
    False.
  - A task gets its coroutine's result from a wrapper coroutine that awaits
    it (`Task._drive`), not from `StopIteration.value`, whose type a static
    program cannot know.
  - `gather` takes coroutines of one result type and returns a list.
    `return_exceptions`, `shield`, `wait`, `as_completed`, `wait_for`,
    `timeout`, task groups, streams, subprocesses, executors and
    thread-safe calls are not ported; neither are contextvars in callbacks,
    debug mode, exception handlers or loop policies.
  - `sleep(delay)` takes no `result`.
  - The loop has no `create_future()`: construct `Future[T]()`. A generic
    class specialized before its base class is declared misses the base's
    fields, and the loop is declared before the futures it would make.
"""

from time import monotonic as _monotonic, sleep as _sleep_blocking
from types import CoroutineType
from typing import Callable, Generator

__all__ = [
    "CancelledError", "InvalidStateError", "AbstractEventLoop", "Future",
    "Task", "get_event_loop", "get_running_loop", "new_event_loop",
    "set_event_loop", "run", "sleep", "create_task", "gather",
]


class CancelledError(BaseException):
    """The Future or Task was cancelled."""


class InvalidStateError(Exception):
    """The operation is not allowed in this state."""


class _TimerHandle:
    def __init__(self, when: float, seq: int, callback: Callable[[], None]) -> None:
        self.when = when
        self.seq = seq
        self.callback = callback
        self.cancelled = False

    def before(self, other: "_TimerHandle") -> bool:
        if self.when != other.when:
            return self.when < other.when
        return self.seq < other.seq

    def cancel(self) -> None:
        self.cancelled = True


class AbstractEventLoop:
    """The event loop: a ready queue of callbacks and a schedule of timers.

    `_run_once` runs every callback that was ready when it started, after
    moving the timers that are due onto the ready queue; with nothing ready it
    sleeps until the first timer, as BaseEventLoop._run_once does.
    """

    def __init__(self) -> None:
        self._ready: list[Callable[[], None]] = []
        self._scheduled: list[_TimerHandle] = []
        self._seq = 0
        self._running = False
        self._stopping = False
        self._closed = False

    def time(self) -> float:
        return _monotonic()

    def call_soon(self, callback: Callable[[], None]) -> None:
        self._check_closed()
        self._ready.append(callback)

    def call_later(self, delay: float, callback: Callable[[], None]) -> _TimerHandle:
        return self.call_at(self.time() + delay, callback)

    def call_at(self, when: float, callback: Callable[[], None]) -> _TimerHandle:
        self._check_closed()
        self._seq += 1
        timer = _TimerHandle(when, self._seq, callback)
        index = len(self._scheduled)
        while index > 0 and timer.before(self._scheduled[index - 1]):
            index -= 1
        self._scheduled.insert(index, timer)
        return timer

    def is_running(self) -> bool:
        return self._running

    def is_closed(self) -> bool:
        return self._closed

    def stop(self) -> None:
        self._stopping = True

    def close(self) -> None:
        if self._running:
            raise RuntimeError("Cannot close a running event loop")
        self._closed = True
        self._ready = []
        self._scheduled = []

    def run_forever(self) -> None:
        self._enter()
        try:
            while not self._stopping:
                self._run_once()
        finally:
            self._stopping = False
            self._leave()

    def _run_until(self, waiter: "_Waiter") -> None:
        self._enter()
        try:
            while not waiter.done():
                self._run_once()
        finally:
            self._leave()

    def _enter(self) -> None:
        self._check_closed()
        if self._running:
            raise RuntimeError("This event loop is already running")
        if _running_loop:
            raise RuntimeError(
                "Cannot run the event loop while another loop is running")
        self._running = True
        _running_loop.append(self)

    def _leave(self) -> None:
        self._running = False
        _running_loop.clear()

    def _check_closed(self) -> None:
        if self._closed:
            raise RuntimeError("Event loop is closed")

    def _run_once(self) -> None:
        if not self._ready and self._scheduled:
            first = self._scheduled[0]
            wait = first.when - self.time()
            if wait > 0:
                _sleep_blocking(wait)
        now = self.time()
        while self._scheduled and self._scheduled[0].when <= now:
            timer = self._scheduled.pop(0)
            if not timer.cancelled:
                self._ready.append(timer.callback)
        batch = self._ready
        self._ready = []
        for callback in batch:
            callback()


# ⛔ One-element lists rather than `AbstractEventLoop | None` globals: a
# module global written from a function needs a concrete type.
_running_loop: list[AbstractEventLoop] = []
_event_loop: list[AbstractEventLoop] = []


def get_running_loop() -> AbstractEventLoop:
    if not _running_loop:
        raise RuntimeError("no running event loop")
    return _running_loop[0]


def new_event_loop() -> AbstractEventLoop:
    return AbstractEventLoop()


def set_event_loop(loop: AbstractEventLoop | None) -> None:
    _event_loop.clear()
    if loop is not None:
        _event_loop.append(loop)


def get_event_loop() -> AbstractEventLoop:
    if _running_loop:
        return _running_loop[0]
    if not _event_loop:
        _event_loop.append(new_event_loop())
    return _event_loop[0]


class _Waiter:
    """What a task waits on: a result that arrives later and wakes callbacks.

    The state and callbacks of CPython's Future, without the result, which is
    typed in Future and Task.
    """

    def __init__(self, is_task: bool) -> None:
        self._state = "PENDING"
        self._callbacks: list[Callable[[], None]] = []
        self._exception: BaseException | None = None
        self._cancel_message = ""
        # A task's cancel() is a request its next step delivers into the
        # coroutine (Task.cancel); a future's takes effect at once.
        # ⛔ A flag rather than an override in Task: a call through a
        # `_Waiter` would then dispatch over the generic subclass, which the
        # dispatch cannot enumerate the instantiations of.
        self._is_task = is_task
        self._must_cancel = False

    def done(self) -> bool:
        return self._state != "PENDING"

    def cancelled(self) -> bool:
        return self._state == "CANCELLED"

    def cancel(self, msg: str = "") -> bool:
        if self._state != "PENDING":
            return False
        if self._is_task:
            self._must_cancel = True
            self._cancel_message = msg
            return True
        return self._set_cancelled(msg)

    def _set_cancelled(self, msg: str) -> bool:
        if self._state != "PENDING":
            return False
        self._state = "CANCELLED"
        self._cancel_message = msg
        self._schedule_callbacks()
        return True

    def exception(self) -> BaseException | None:
        if self._state == "CANCELLED":
            raise CancelledError(self._cancel_message)
        if self._state != "FINISHED":
            raise InvalidStateError("Exception is not set.")
        return self._exception

    def set_exception(self, exception: BaseException) -> None:
        if self._state != "PENDING":
            raise InvalidStateError("invalid state")
        self._exception = exception
        self._state = "FINISHED"
        self._schedule_callbacks()

    def _wake(self, callback: Callable[[], None]) -> None:
        if self._state != "PENDING":
            get_event_loop().call_soon(callback)
        else:
            self._callbacks.append(callback)

    def _schedule_callbacks(self) -> None:
        callbacks = self._callbacks
        self._callbacks = []
        loop = get_event_loop()
        for callback in callbacks:
            loop.call_soon(callback)

    def _check_result(self) -> None:
        if self._state == "CANCELLED":
            raise CancelledError(self._cancel_message)
        if self._state != "FINISHED":
            raise InvalidStateError("Result is not ready.")
        exc = self._exception
        if exc is not None:
            raise exc


class Future[T](_Waiter):
    """A result that arrives later (futures.Future)."""

    def __init__(self) -> None:
        super().__init__(False)
        self._result: list[T] = []

    def result(self) -> T:
        self._check_result()
        return self._result[0]

    def set_result(self, result: T) -> None:
        if self._state != "PENDING":
            raise InvalidStateError("invalid state")
        self._result.append(result)
        self._state = "FINISHED"
        self._schedule_callbacks()

    def add_done_callback(self, fn: Callable[["Future[T]"], None]) -> None:
        self._wake(lambda: fn(self))

    def __await__(self) -> Generator[object, None, T]:
        if not self.done():
            yield self
        if not self.done():
            raise RuntimeError("await wasn't used with future")
        return self.result()


class Task[T](_Waiter):
    """A coroutine scheduled on the loop (tasks.Task).

    `_step` resumes the coroutine once; what it yields is the waiter it waits
    on, and the task resumes when that waiter is done. None gives up the turn
    and is resumed on the next one.
    """

    def __init__(self, coro: CoroutineType[object, None, T]) -> None:
        super().__init__(True)
        self._result: list[T] = []
        self._coro = coro
        self._driver: CoroutineType[object, None, None] = self._drive()
        get_event_loop().call_soon(self._step)

    async def _drive(self) -> None:
        value = await self._coro
        self._result.append(value)

    def result(self) -> T:
        self._check_result()
        return self._result[0]

    def add_done_callback(self, fn: Callable[["Task[T]"], None]) -> None:
        self._wake(lambda: fn(self))

    def _step(self) -> None:
        if self.done():
            return
        # ⛔ What the coroutine yielded is handled inside the `try`, not after
        # it as CPython does: a value assigned in a `try` and read after it
        # lives in a cell, and a cell cannot hold an `object`. Nothing the
        # handling calls raises.
        try:
            if self._must_cancel:
                self._must_cancel = False
                yielded = self._driver.throw(CancelledError(self._cancel_message))
            else:
                yielded = self._driver.send(None)
            if isinstance(yielded, _Waiter):
                if self._must_cancel:
                    yielded.cancel(self._cancel_message)
                yielded._wake(self._step)
            else:
                get_event_loop().call_soon(self._step)
        except StopIteration:
            self._state = "FINISHED"
            self._schedule_callbacks()
        except CancelledError:
            self._set_cancelled(self._cancel_message)
        except BaseException as exc:
            self.set_exception(exc)

    def __await__(self) -> Generator[object, None, T]:
        if not self.done():
            yield self
        if not self.done():
            raise RuntimeError("await wasn't used with future")
        return self.result()


class _Yield:
    """`sleep(0)`'s bare yield: give up the turn once (tasks.__sleep0)."""

    def __await__(self) -> Generator[object, None, None]:
        yield None


def create_task[T](coro: CoroutineType[object, None, T]) -> Task[T]:
    get_running_loop()
    return Task[T](coro)


async def sleep(delay: float) -> None:
    if delay <= 0:
        await _Yield()
        return
    future = Future[None]()
    timer = get_running_loop().call_later(delay, lambda: future.set_result(None))
    try:
        await future
    finally:
        timer.cancel()


async def gather[T](*coros: CoroutineType[object, None, T]) -> list[T]:
    tasks = [create_task(coro) for coro in coros]
    results: list[T] = []
    for task in tasks:
        results.append(await task)
    return results


def run[T](main: CoroutineType[object, None, T]) -> T:
    if _running_loop:
        raise RuntimeError(
            "asyncio.run() cannot be called from a running event loop")
    loop = new_event_loop()
    set_event_loop(loop)
    try:
        task = Task[T](main)
        loop._run_until(task)
        return task.result()
    finally:
        set_event_loop(None)
        loop.close()
