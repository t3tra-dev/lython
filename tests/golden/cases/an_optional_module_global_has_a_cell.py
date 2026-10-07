# WHAT: a module global annotated with a union and used by a function has a
# cell: functions read it, `global` writes reach it, `if _cache is None:
# _cache = {...}` is the lazy cache CPython code writes, a guard or a store
# narrows it for the code after, a call that rebinds it takes the proof away,
# a list held in it grows in place through `append`, and a linked list hung
# off it is pushed and popped. Module code reads the same cell.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: every one of these was
# refused or crashed only once the program ran -- a read that freed what the
# cell held, a mutation that wrote a list's lanes where the cell keeps a tag
# -- so the values printed and a clean exit are the claim.

_cache: dict[str, int] | None = None
name: str | None = None
count: int | None = 3
def get_cache() -> dict[str, int]:
    global _cache
    if _cache is None:
        print("build")
        _cache = {"a": 1}
    return _cache
def show() -> None:
    print(name, count)
    if count is not None:
        print(count + 1)
print(get_cache(), get_cache())
show()
name = "x"
count = None
show()
def setname(v: str | None) -> None:
    global name
    name = v
setname("y")
show()
setname(None)
show()
print(name is None, count)
if name is None:
    print("module narrowing")
class Node:
    def __init__(self, v: int, nxt: "Node | None") -> None:
        self.v = v
        self.nxt = nxt
head: Node | None = None
def push(v: int) -> None:
    global head
    head = Node(v, head)
def walk() -> list[int]:
    out: list[int] = []
    cur = head
    while cur is not None:
        out.append(cur.v)
        cur = cur.nxt
    return out
for i in range(5):
    push(i)
print(walk())
def pop() -> int:
    global head
    if head is None:
        return -1
    v = head.v
    head = head.nxt
    return v
print(pop(), pop(), walk())
mixed: int | str | None = 1
def bump() -> None:
    global mixed
    if isinstance(mixed, int):
        mixed += 10
    elif isinstance(mixed, str):
        mixed = mixed + "!"
    else:
        mixed = 0
for _ in range(2):
    bump()
print(mixed)
mixed = "s"
bump()
print(mixed)
mixed = None
bump()
print(mixed)
if isinstance(mixed, int):
    print(mixed + 1)
buf: list[str] | None = None
def add(s: str) -> None:
    global buf
    if buf is None:
        buf = []
    buf.append(s)
for i in range(100):
    add(str(i))
    if i % 10 == 9:
        print(len(buf) if buf is not None else 0, end=" ")
        buf = None
print()
def stale() -> None:
    if buf is None:
        print("none")
stale()
