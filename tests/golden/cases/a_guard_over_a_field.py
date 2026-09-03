# `if self.left is None:` proves something about a FIELD, and the machinery
# only knew names -- so the binary-search-tree idiom, and every linked
# structure with it, was refused: "union<Tree, None> does not provide manifest
# method 'insert'". `v = self.left; if v is not None:` -- the same program with
# the read bound first -- has always compiled.


class Tree:
    def __init__(self, v: int) -> None:
        self.v = v
        self.left = None
        self.right = None

    def insert(self, v: int) -> None:
        if v < self.v:
            if self.left is None:
                self.left = Tree(v)
            else:
                self.left.insert(v)
        else:
            if self.right is None:
                self.right = Tree(v)
            else:
                self.right.insert(v)

    def walk(self) -> list[int]:
        out: list[int] = []
        if self.left is not None:
            out += self.left.walk()
        out.append(self.v)
        if self.right is not None:
            out += self.right.walk()
        return out

    # The early-return spelling of the same guard: the fact survives the `if`
    # because only one side of it reaches the line below.
    def depth(self) -> int:
        if self.left is None:
            return 1
        return 1 + self.left.depth()


t = Tree(5)
for value in [3, 8, 1, 4, 9]:
    t.insert(value)
print(t.walk())
print(t.depth())


class Box:
    def __init__(self) -> None:
        self.v = None

    def set(self, s: str) -> None:
        self.v = s

    # Two reads under one guard, and a conditional expression spelling.
    def shout(self) -> str:
        if self.v is not None:
            return self.v.upper() + self.v.lower()
        return "-"

    def sized(self) -> str:
        return str(len(self.v)) if self.v is not None else "-"


b = Box()
print(b.shout(), b.sized())
b.set("Ab")
print(b.shout(), b.sized())
