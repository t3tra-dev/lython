# What: `and` / `or` with an object operand, inside a generator. The generator
# threads its locals through the short-circuit's merge, beside the merge's own
# value -- which, on the edge where the operand decides, IS that local. The
# token went to the carried local while the verifier followed it into the
# value, so every one of these was refused ("still owned when a call may
# unwind"); the same functions not written as generators always compiled.
#
# Why execution: the refusal is gone either way a fix could go, and only the
# values -- and the leak gate this case is a member of -- say the local's one
# reference went to exactly one place.
from typing import Iterator


def blocks(lines: list[str]) -> Iterator[tuple[str, list[str]]]:
    kind = "para"
    buf: list[str] = []
    for line in lines:
        if line.startswith("-"):
            if kind != "list" and buf:
                yield (kind, buf)
                buf = []
            kind = "list"
        buf.append(line)
    if buf:
        yield (kind, buf)


def guarded(c: bool) -> Iterator[int]:
    s = "x" + str(c)
    if c and s:
        yield 1
    yield len(s)


def as_a_value(c: bool) -> Iterator[int]:
    buf: list[int] = [1, 2]
    x = c and buf
    print(x)
    yield len(buf)


def either(c: bool) -> Iterator[int]:
    buf: list[int] = [3]
    if c or buf:
        yield 1
    yield len(buf)


print(list(blocks(["a", "-b", "c", "-d"])))
print(list(guarded(True)), list(guarded(False)))
print(list(as_a_value(True)), list(as_a_value(False)))
print(list(either(True)), list(either(False)))
