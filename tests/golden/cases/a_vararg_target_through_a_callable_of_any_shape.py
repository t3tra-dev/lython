# WHAT: a function with `*args` stored as a `Callable[P, object]` beside
# others and called through `lambda: function(*args, **kwargs)` -- the shape
# of unittest's addModuleCleanup.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which target the stored
# value holds, and so which arm of the call runs, is decided at run time.
#
# ⛔ The arm for a `*args` target built its argument tuple and then wrote the
# rest of itself after the call's merge point -- "block with no terminator".
from collections.abc import Callable

cleanups: list[Callable[[], object]] = []


def add_cleanup[**P](function: Callable[P, object], /, *args: P.args,
                     **kwargs: P.kwargs) -> None:
    cleanups.append(lambda: function(*args, **kwargs))


def joined(*parts: object) -> str:
    return "+".join([str(part) for part in parts])


def upper(text: str) -> str:
    return text.upper()


events: list[str] = []
add_cleanup(events.append, "appended")
add_cleanup(upper, "x")
add_cleanup(joined, "a")
while cleanups:
    print(cleanups.pop()())
print(events)
