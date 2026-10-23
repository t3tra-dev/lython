"""Utilities for with-statement contexts.

A port of CPython 3.14's Lib/contextlib.py, limited to `nullcontext`.

Deviations from CPython:
- `contextmanager` is not defined here: the compiler rewrites a decorated
  generator into its context manager where the decorator is spelled
  (EmitterWith.cpp), because the generator's frame is the object it returns.
- `nullcontext` is generic in its `enter_result` rather than subclassing
  `AbstractContextManager`; it has no `__aenter__` / `__aexit__`.
"""


class nullcontext[T]:
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager.
    """

    enter_result: T

    def __init__(self, enter_result: T = None) -> None:
        self.enter_result = enter_result

    def __enter__(self) -> T:
        return self.enter_result

    def __exit__(self, *excinfo: object) -> None:
        pass
