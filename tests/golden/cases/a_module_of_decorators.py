# WHAT: a module the next case imports -- a decorator, a decorator factory
# and one that records each call it wraps. Run alone it prints nothing.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: it is the imported half of
# a_decorator_is_any_function_the_program_names.py, which needs it on disk;
# running it alone keeps it a program of its own.
from typing import Callable

calls: list[str] = []


def twice(f: Callable[[int], int]) -> Callable[[int], int]:
    def wrapper(n: int) -> int:
        return f(f(n))
    return wrapper


def times(k: int) -> Callable[[Callable[[int], int]], Callable[[int], int]]:
    def decorate(f: Callable[[int], int]) -> Callable[[int], int]:
        def wrapper(n: int) -> int:
            return f(n) * k
        return wrapper
    return decorate


def logged(f: Callable[[int], int]) -> Callable[[int], int]:
    def wrapper(n: int) -> int:
        calls.append("call " + str(n))
        return f(n)
    return wrapper
