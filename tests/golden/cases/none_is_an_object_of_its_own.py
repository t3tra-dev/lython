# WHAT: None is an object of its own -- in a list[object], a dict value, a
#   union element, a slice bound and a tuple it compares, hashes, prints and
#   tests as CPython's None, and hash(None) is CPython's constant.
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the None word is written
#   by one lowering and read by another (the hooks, slice, identity), so only
#   the printed values show that the two agree.
class Node:
    def __init__(self, v: int, nxt: "Node | None") -> None:
        self.v = v
        self.nxt = nxt

def walk(n: "Node | None") -> list[int]:
    out: list[int] = []
    while n is not None:
        out.append(n.v)
        n = n.nxt
    return out

xs: list[object] = [None, 1, "a", None]
print(xs, xs[0] is None, xs[0] == None, xs.count(None), None in xs)
d: dict[str, int | None] = {"a": None, "b": 2}
print(d, d["a"] is None, d.get("c"), list(d.values()))
u: list[int | None] = [1, None, 3]
print(u, [x for x in u if x is not None], u.index(None))
print(walk(Node(1, Node(2, None))), walk(None))
s = slice(None, 5, None)
print(s, s.start is None, [0, 1, 2, 3, 4, 5, 6][s], slice(2, None).indices(10))
t = (None, 1)
print(t, t[0] is None, hash(None) == hash(t[0]), {None: 1}[None], {None, None})
def f(o: object) -> bool:
    return o is None
print(f(None), f(0), f(xs[3]), repr(None), str(None))
opt: list["Node | None"] = [None, Node(5, None)]
n = opt[1]
print([o is None for o in opt], n.v if n is not None else -1)
print(hash(None))
