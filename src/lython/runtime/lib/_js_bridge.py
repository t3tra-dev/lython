"""The program's half of a JavaScript callback (docs/js-host.md).

A Python callable handed to the host is wrapped, where it is handed over, in
a closure of no arguments that reads the host's arguments and leaves its
result with the host. This module keeps those closures alive by slot until
the host's function is collected, and its `dispatch` is the one Python
function the host calls: lyc gives it the C entry point LyJs_Dispatch.

An exception a callback raises does not unwind into the host -- it is caught
here and the host's function throws an Error with its message, which is what
the host can make of it.
"""

from typing import Callable

from _js import JsProxy, callback_slot, fail_callback, function_for

_callbacks: dict[int, Callable[[], None]] = {}
_last_slot: list[int] = [0]


def wrap(run: Callable[[], None]) -> JsProxy:
    _last_slot[0] += 1
    slot = _last_slot[0]
    _callbacks[slot] = run
    return function_for(slot)


def dispatch() -> None:
    try:
        _callbacks[callback_slot()]()
    except BaseException as error:
        fail_callback(type(error).__name__ + ": " + str(error))


def release() -> None:
    del _callbacks[callback_slot()]
