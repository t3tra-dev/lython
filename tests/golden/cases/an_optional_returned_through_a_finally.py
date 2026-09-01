# What: a union return through a finally needs a value on the path that did NOT
# return, and the payload is lanes plus a TAG -- so the default has to name an
# active member. Only running it shows the value each path actually carried
# out, and that the finally ran on both.
def pick(flag: int) -> "int | None":
    try:
        if flag > 0:
            return 7
        return None
    finally:
        print("checked")


print(pick(1), pick(-1))


def find(xs: "list[str]", needle: str) -> "str | None":
    try:
        for x in xs:
            if x == needle:
                return x
        return None
    finally:
        print("searched")


print(find(["a", "b"], "b"), find(["a"], "z"))


class Box:
    def __init__(self, n: int) -> None:
        self.n = n


def maybe(flag: int) -> "Box | None":
    try:
        if flag > 0:
            return Box(flag)
        return None
    finally:
        print("built")


got = maybe(3)
print(got.n if got is not None else -1, maybe(0))


class Guard:
    def __enter__(self) -> "Guard":
        return self

    def __exit__(self, a: object, b: object, c: object) -> bool:
        print("closed")
        return False


def through_a_with(flag: int) -> "int | None":
    with Guard():
        if flag > 0:
            return flag * 2
        return None


print(through_a_with(2), through_a_with(-1))
