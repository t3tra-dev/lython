# Why execution: the annotation decides which branch narrowing keeps, so only
# running it shows the string form resolved to the same union the unquoted
# form does. `"int | None"` was refused -- "string annotation is not a simple
# class name" -- while `int | None` one line away resolved. Every class a
# string annotation can name is predeclared before bodies are typed, which is
# what makes the simple-name case need no second pass, and splitting on `|`
# does not change that.
class Leaf:
    def __init__(self, v: int) -> None:
        self.v = v


def show(v: "int | None") -> str:
    if v is None:
        return "none"
    return str(v)


def pick(flag: bool) -> "int | None":
    if flag:
        return 7
    return None


def unquoted(v: int | None) -> str:
    if v is None:
        return "none"
    return str(v)


def leafy(flag: bool) -> "Leaf | None":
    if flag:
        return Leaf(3)
    return None


def main() -> None:
    print(show(1), show(None), show(pick(True)), show(pick(False)))
    print(unquoted(2), unquoted(None))
    print(leafy(False) is None, leafy(True) is None)
    label: "str | None" = None
    print(label is None)
    label = "set"
    print(label)


main()
