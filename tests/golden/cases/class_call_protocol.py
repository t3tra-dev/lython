# Why execution: the value is the evidence the class's own __call__ ran.
# `v(2)` over a class that defines __call__ died in the lowering as "runtime
# manifest has no V.__call__ method" -- py.call resolves its target against
# the manifest, and a source class is not in it, the same repair __iter__ and
# the unary dunders needed.
class Adder:
    def __init__(self, base: int) -> None:
        self.base = base

    def __call__(self, v: int) -> int:
        return self.base + v


class Greet:
    def __call__(self, who: str, punct: str = "!") -> str:
        return "hi " + who + punct


def main() -> None:
    a = Adder(10)
    print(a(1), a(2), Adder(100)(5))
    g = Greet()
    print(g("x"), g("y", "?"))
    total = 0
    for i in range(3):
        total = a(total)
    print(total)


main()
