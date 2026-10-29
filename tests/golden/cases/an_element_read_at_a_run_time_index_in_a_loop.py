# Why execution: the leak gate measures this program (tests/CMakeLists.txt),
# and only a run can count what each call leaves behind.
#
# A tuple display read at a run-time index inside a loop selects its element
# with one arm per candidate, and the read takes a reference of its own. The
# first candidate's own reference was released through the selection --
# after the read, inside the loop, where it released whichever candidate was
# read -- so it was never released at all: one object per call, for every
# element type, and for a union of them the selection did not lower.
class A:
    def __init__(self, n: int) -> None:
        self.n = n


def same(k: int) -> int:
    r = (A(k), A(k + 1), A(k + 2))
    t = 0
    for i in range(3):
        t += r[i].n
    return t


def mixed(k: int) -> list[A | str]:
    r = (A(k), str(k) + "y")
    out: list[A | str] = []
    for i in range(2):
        out.append(r[i])
    return out


def chosen(k: int) -> A | str:
    r = (A(k), str(k) + "x")
    return r[k % 2]


total = 0
for j in range(40):
    total += same(j)
    for w in mixed(j):
        total += w.n if isinstance(w, A) else len(w)
    v = chosen(j)
    total += v.n if isinstance(v, A) else len(v)
print(total)
