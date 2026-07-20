class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"


class OnlyStr:
    def __init__(self, x: int) -> None:
        self.x = x

    def __str__(self) -> str:
        return f"S{self.x}"


class OnlyRepr:
    def __init__(self, x: int) -> None:
        self.x = x

    def __repr__(self) -> str:
        return f"R{self.x}"


class MyError(Exception):
    pass


class LoudError(Exception):
    def __str__(self) -> str:
        return "LOUD"


# print renders through str(): __str__ outranks __repr__.
p = Point(1, 2)
print(p)
print(str(p))
print(repr(p))
print(p, repr(p))

# A class with only __str__ prints through it; only __repr__ serves both.
a = OnlyStr(3)
print(a)
print(str(a))
b = OnlyRepr(4)
print(b)
print(str(b))
print(a, b)

# Exceptions: str is the message, repr is ClassName(...); a user __str__
# overrides the message form.
e1 = ValueError("boom")
print(e1)
print(str(e1), repr(e1))
e2 = MyError("custom")
print(e2)
print(str(e2), repr(e2))
e3 = LoudError("quiet")
print(e3)
print(str(e3))

# str(x) == repr(x) for the non-str builtins; str(s) is the identity.
print(str(5), str(3.5), str(True), str(None))
print(str("abc"))
print(str([1, 2]), str((1, "a")), str({"k": 1}))
