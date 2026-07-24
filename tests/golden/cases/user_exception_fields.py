# R2: a user exception class declares its own instance fields. The exception
# object's layout is fixed by the taxonomy (3-word header + message), so the
# fields live in a [count, count x box16] block hung off extended header word
# 4 -- reached only through the BaseException payload primitives, and released
# with the exception. Field stores must precede super().__init__(...): the
# message argument's ownership transfers into the exception there, so a store
# reading the same value afterwards is a loud use-after-release.
class DecodeError(ValueError):
    def __init__(self, msg: str, doc: str, pos: int) -> None:
        self.msg = msg
        self.doc = doc
        self.pos = pos
        self.lineno = doc.count("\n", 0, pos) + 1
        super().__init__(msg + ": line " + str(self.lineno) + " (char " + str(pos) + ")")


def parse(text: str) -> int:
    if text != "1":
        raise DecodeError("Expecting value", text, 0)
    return 1


try:
    print(parse("x"))
except DecodeError as e:
    print(str(e))
    print(e.msg)
    print(e.doc)
    print(e.pos)
    print(e.lineno)
except ValueError:
    print("value error")

# The block is shared mutable state reachable from the handler binding: a
# rebind through the caught exception is visible to every later read.
try:
    raise DecodeError("first", "ab\ncd", 4)
except DecodeError as err:
    print(err.msg)
    print(err.lineno)
    err.msg = "second"
    print(err.msg)
    print(len(err.msg))



# A field's declared type also resolves through a local bound earlier in
# __init__ -- how CPython's own JSONDecodeError computes lineno/colno.
class PositionError(ValueError):
    def __init__(self, msg: str, doc: str, pos: int) -> None:
        lineno = doc.count("\n", 0, pos) + 1
        colno = pos - doc.rfind("\n", 0, pos)
        self.msg = msg
        self.doc = doc
        self.pos = pos
        self.lineno = lineno
        self.colno = colno
        super().__init__(
            "%s: line %d column %d (char %d)" % (msg, lineno, colno, pos)
        )


try:
    raise PositionError("Expecting value", "{\n  bad}", 4)
except PositionError as pe:
    print(str(pe))
    print(pe.pos)
    print(pe.lineno)
    print(pe.colno)


# Field reads escaping their owner keep the payload alive.
def message_of(text: str) -> str:
    try:
        parse(text)
    except DecodeError as caught:
        return caught.msg
    return "ok"


print(message_of("x"))
print(message_of("1"))
