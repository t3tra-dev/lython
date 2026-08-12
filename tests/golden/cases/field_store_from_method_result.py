# Why execution: the values are what the store has to preserve, and the store
# was refused outright. A manifest method's result contract is a NAME, and
# rebuilding a type from a name drops the type arguments -- so `xs.copy()`
# arrived at a declared `list[int]` field as a bare `builtins.list`:
# "attribute value builtins.list is not assignable to field
# builtins.list<int>". Same for `xs + ys` and `xs * n`, whose manifest result
# contracts are also bare names, and the local-variable spelling of each hid
# it because the annotation coerced.
class Grid:
    def __init__(self, w: int, h: int) -> None:
        self.w = w
        self.cells: list[int] = [0] * (w * h)

    def set(self, x: int, y: int, v: int) -> None:
        self.cells[y * self.w + x] = v

    def get(self, x: int, y: int) -> int:
        return self.cells[y * self.w + x]


class Holder:
    def __init__(self, xs: list[int]) -> None:
        self.copied: list[int] = xs.copy()
        self.joined: list[int] = xs + [9]
        self.doubled: list[int] = xs * 2
        self.text: str = ",".join(["a", "b"])


def main() -> None:
    g = Grid(2, 2)
    g.set(1, 1, 7)
    print(g.get(1, 1), g.get(0, 0), sum(g.cells), len(g.cells))
    h = Holder([1, 2])
    print(h.copied, h.joined, h.doubled, h.text)
    h.copied.append(3)
    print(h.copied, h.joined)


main()
