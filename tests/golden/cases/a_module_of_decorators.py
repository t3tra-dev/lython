# WHAT: a module other cases import -- a decorator, a decorator factory, one
# that records each call it wraps, and a generic one that records being
# applied. Run alone it prints nothing.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: it is the imported half of
# a_decorator_is_any_function_the_program_names.py and
# a_decorated_method_is_what_its_decorators_return.py, which need it on disk;
# running it alone keeps it a program of its own.
from typing import Callable, TypeVar

F = TypeVar("F")

calls: list[str] = []
applied: list[str] = []


def mark(f: F) -> F:
    applied.append("mark")
    return f


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
