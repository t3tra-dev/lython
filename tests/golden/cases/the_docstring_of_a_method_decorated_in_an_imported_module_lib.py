# The class for cases/the_docstring_of_a_method_decorated_in_an_imported_module.py.
from collections.abc import Callable


def keep[T](f: Callable[[T], str]) -> Callable[[T], str]:
    return f


def wrap[T](f: Callable[[T], str]) -> Callable[[T], str]:
    def wrapper(self: T) -> str:
        """The wrapper's own words."""
        return "wrapped " + f(self)
    return wrapper


class Thing:
    @keep
    def kept(self) -> str:
        """Kept as written."""
        return "kept"

    @wrap
    def wrapped(self) -> str:
        """Hidden by the wrapper."""
        return "inner"

    def plain(self) -> str:
        """Plain."""
        return "plain"
