# What: a union parameter has no single runtime contract, so its default has
# to be built at the contract the literal itself names and then wrapped into
# the union's lanes. Only the omitted-argument call shows whether the value
# that arrives is the default or whatever the lanes happened to hold.
def label(x: "int | str" = 3) -> "int | str":
    return x


print(label("a"))
print(label())


def greet(name: "str | None" = "world") -> str:
    if name is None:
        return "nobody"
    return "hello " + name


print(greet(None), greet(), greet("you"))


def count(n: "int | None" = 2) -> int:
    if n is None:
        return 0
    return n * 10


print(count(None), count(), count(5))


def ratio(r: "float | None" = 0.5) -> float:
    if r is None:
        return 0.0
    return r + 1.0


print(ratio(None), ratio())


def flag(b: "bool | None" = True) -> str:
    if b is None:
        return "unset"
    return "on" if b else "off"


print(flag(None), flag(), flag(False))


def raw(data: "bytes | None" = b"xy") -> int:
    if data is None:
        return -1
    return len(data)


print(raw(None), raw(), raw(b"abc"))
