# A field's declared type comes from the whole expression that initializes it,
# not only from a bare parameter or a bare local. `self.n = pos + 1` used to
# reach inferExpr with `pos` unbound, so the field typed as `object` -- and an
# object-typed field is not a diagnostic, it is a field whose reads answer None
# and whose value silently fails to survive a carry.
class Plain:
    def __init__(self, pos: int) -> None:
        self.pos = pos
        self.next = pos + 1
        self.label = "p" + str(pos)
        doubled = pos * 2
        self.doubled = doubled
        self.tripled = doubled + pos


p = Plain(4)
print(p.pos, p.next, p.doubled, p.tripled)
print(p.label)
print(p.next + 10)
print(len(p.label))


# The same through an exception class, whose fields live in the word[4] block.
class Where(ValueError):
    def __init__(self, msg: str, doc: str, pos: int) -> None:
        lineno = doc.count("\n", 0, pos) + 1
        self.msg = msg
        self.pos = pos
        self.lineno = lineno
        self.colno = pos - doc.rfind("\n", 0, pos)
        self.tag = msg + "@" + str(pos)
        super().__init__(msg)


try:
    raise Where("bad", "ab\ncdef", 5)
except Where as e:
    print(e.msg, e.pos, e.lineno, e.colno)
    print(e.tag)
    print(e.lineno + e.colno)


# An explicit annotation on the attribute still wins, and mixes with inferred
# siblings in the same field block.
class Annotated:
    def __init__(self, n: int) -> None:
        raw = n + 1
        self.count: int = raw
        self.name = "n" + str(raw)


a = Annotated(2)
print(a.count, a.name)
print(a.count * 3)
