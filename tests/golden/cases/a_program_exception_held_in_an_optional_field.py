# WHAT: a program's exception class -- plain, and one with fields of its own
# -- stored in a `T | None` field by a context manager's `__exit__` (the
# shape of unittest's assertRaises), on the with's exception exit, and read
# back after the block; and the same class stored by a plain assignment.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the field was laid out
# inline because the class has no manifest shape of its own; the with's
# second inlined `__exit__` then released the lanes the first one stored
# ("operand #0 does not dominate this use"), and an older build printed
# None where the exception was. Which exception was kept is seen in what it
# says.
class ParseError(Exception):
    def __init__(self, msg: str, pos: int) -> None:
        self.msg = msg
        self._pos = pos
        super().__init__(msg)

    @property
    def pos(self) -> int | None:
        return None if self._pos < 0 else self._pos


class Plain(Exception):
    pass


class Catch[E: BaseException]:
    def __init__(self, expected: type[E]) -> None:
        self.expected = expected
        self._exception: E | None = None

    def __enter__(self) -> "Catch[E]":
        return self

    def __exit__(self, t: type[BaseException] | None, v: BaseException | None,
                 tb: object) -> bool:
        if isinstance(v, self.expected):
            self._exception = v
            return True
        return False

    def caught(self) -> E | None:
        return self._exception


class Holder:
    def __init__(self) -> None:
        self.last: Plain | None = None


def parse(text: str) -> None:
    raise ParseError("bad " + text, len(text))


with Catch(ParseError) as cm:
    parse("(a")
got = cm.caught()
if got is not None:
    print(got.msg, got.pos, str(got))
with Catch(Plain) as quiet:
    pass
print(quiet.caught())
h = Holder()
try:
    raise Plain("plain")
except Plain as e:
    h.last = e
print(h.last)
