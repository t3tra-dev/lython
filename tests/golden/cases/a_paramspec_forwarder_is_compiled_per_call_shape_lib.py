from collections.abc import Callable


def call[**P, R](f: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    return f(*args, **kwargs)


def twice[**P](f: Callable[P, int], *args: P.args, **kwargs: P.kwargs) -> int:
    return call(f, *args, **kwargs) + call(f, *args, **kwargs)
