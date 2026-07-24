# Cross-track: a user exception's own fields (wave25/abi, extended header
# word[4]) read inside an except handler and REBOUND onto locals that the code
# after the try reads (wave25/defects, handler rebinds carried out of a try
# whose body always raises). Neither track could pin this alone: abi's cases
# consume the fields inside the handler, defects' cases carry only values that
# never came out of an exception payload.


class JSONDecodeError(ValueError):
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


def scan(text: str) -> int:
    raise JSONDecodeError("Expecting value", text, 4)


# The try body always raises, so the handler's rebinds are the only lane
# reaching the post-try reads -- and every rebound value is a field load.
def report(text: str) -> str:
    where = "none"
    line = 0
    col = 0
    off = 0
    try:
        scan(text)
    except JSONDecodeError as e:
        where = e.msg
        line = e.lineno
        col = e.colno
        off = e.pos
    return where + " @" + str(line) + ":" + str(col) + " #" + str(off)


print(report("{\n  bad}"))
print(report("abcdefg"))


# The str field escapes the handler AND the exception: `doc` outlives the
# exception object that owned the box.
def doc_of(text: str) -> str:
    kept = ""
    try:
        scan(text)
    except JSONDecodeError as e:
        kept = e.doc
    return kept + "!"


print(doc_of("payload"))
print(len(doc_of("payload")))


# A conditionally-raising body: the fall-through lane and the handler lane both
# contribute, and the handler lane's values come from the field block.
def maybe(text: str, fail: int) -> str:
    label = "clean"
    depth = -1
    try:
        if fail == 1:
            scan(text)
        label = "ok"
        depth = 0
    except JSONDecodeError as e:
        label = e.msg
        depth = e.colno
    return label + "/" + str(depth)


print(maybe("xxxxyy", 1))
print(maybe("xxxxyy", 0))


# Two handlers, the specific one reading fields the base one cannot see.
def by_class(text: str, kind: int) -> str:
    out = "start"
    try:
        if kind == 1:
            scan(text)
        raise ValueError("plain")
    except JSONDecodeError as e:
        out = "decode:" + e.msg + ":" + str(e.pos)
    except ValueError as e:
        out = "value:" + str(e)
    return out


print(by_class("zzzzz", 1))
print(by_class("zzzzz", 2))


# Module level: the same carry, with the handler binding mutated first so the
# carried local reads the REBOUND field rather than the original.
tag = "before"
line_no = 0
try:
    raise JSONDecodeError("Extra data", "ab\ncdef", 5)
except JSONDecodeError as e:
    e.msg = e.msg + " (fixed)"
    tag = e.msg
    line_no = e.lineno
print(tag)
print(line_no)
print(len(tag))
