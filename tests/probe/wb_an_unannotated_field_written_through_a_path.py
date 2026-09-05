# OPEN 2026-09-05. An UNANNOTATED field of one class, written from ANOTHER
# class's method through a field path, keeps whatever its constructor gave it:
#
#     error: attribute value '!py.contract<"Cell">' is not assignable to field
#     '!py.literal<None>'
#
# `collectClassFields` reads `self.<name>` and, since 2026-09-03, a NAME whose
# inferred type is this class (`other.nxt = self`, the way a linked structure
# links). What it does not read is a write whose owner is an ATTRIBUTE of
# another class -- `self.tail.nxt = node` inside `Queue`, which names `Cell.nxt`.
#
# ⭐ THE CROSSINGS, all measured:
#   - `self.head = None` then `self.head = Cell(v)` in the SAME class: works.
#   - `self.n = None` then an int through a method of the same class: works.
#   - `other.nxt = self` through a parameter of this class: works (2026-09-03).
#   - annotating `self.nxt: Optional["Cell"] = None`: works, and is the project's
#     own convention for `lib/*.py`.
#   - this file: refused.
#
# ⛔ NOT a one-line extension of the same walk. Reading `self.tail.nxt = node`
# means knowing `Queue.tail`'s type while walking `Cell`, and `Queue` is declared
# AFTER `Cell` -- its own field walk has not run. That is the declare-then-define
# ordering problem, and the fix has to be a phase split rather than another
# target shape.
class Cell:
    def __init__(self, v: int) -> None:
        self.v: int = v
        self.nxt = None


class Queue:
    def __init__(self) -> None:
        self.head = None
        self.tail = None

    def push(self, v: int) -> None:
        node = Cell(v)
        if self.tail is None:
            self.head = node
            self.tail = node
        else:
            self.tail.nxt = node
            self.tail = node

    def drain(self) -> list[int]:
        out: list[int] = []
        cur = self.head
        while cur is not None:
            out.append(cur.v)
            cur = cur.nxt
        return out


q = Queue()
for i in [1, 2, 3]:
    q.push(i)
print(q.drain())
