# WHAT: `isinstance` where the value's static type says nothing -- an `object`
# parameter, an `object` element -- and the narrowed value is then USED. Also
# the exception arm, where the class the test compares is not the word every
# other object keeps it in.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: every wrong answer here is
# a value. A class test that reads the wrong header word compiles and returns
# False for an object of exactly that class; a narrowed value that aliases the
# box instead of unboxing it reads the entity's refcount as the first field, so
# `o.n` prints 1 for every object. Both read as plausible output.
#
# ⛔ `int` AND `bool` ARE TESTED BUT NOT NARROWED, so the arms below only say
# which they were. A bool has no runtime entity to take a view of, and an `int`
# test accepts a bool -- Python's bool IS an int -- whose box is a different
# object than an int's. The answer is CPython's either way; the value is not
# handed to the branch as that type.
import sys


class Shape:
    n: int

    def __init__(self, n: int) -> None:
        self.n = n

    def area(self) -> int:
        return self.n


class Square(Shape):
    def area(self) -> int:
        return self.n * self.n


class Other:
    pass


class MyErr(Exception):
    pass


def describe(o: object) -> str:
    if isinstance(o, Square):
        return "square " + str(o.n) + " " + str(o.area())
    if isinstance(o, Shape):
        return "shape " + str(o.n) + " " + str(o.area())
    if isinstance(o, Other):
        return "other"
    if isinstance(o, MyErr):
        return "myerr"
    if isinstance(o, ZeroDivisionError):
        return "zde"
    if isinstance(o, Exception):
        return "exception"
    if isinstance(o, str):
        return "str " + o.upper() + " " + str(len(o))
    if isinstance(o, list):
        return "list " + str(len(o))
    if isinstance(o, dict):
        return "dict " + str(len(o))
    if isinstance(o, (tuple, set)):
        return "seq"
    if isinstance(o, float):
        return "float " + str(o * 2)
    if isinstance(o, bool):
        return "bool"
    if isinstance(o, int):
        return "int"
    return "?"


values: "list[object]" = [
    [1, 2],
    {"k": 1},
    (1, 2, 3),
    {9},
    Shape(3),
    Square(3),
    Other(),
    MyErr("m"),
    ZeroDivisionError("z"),
    ValueError("v"),
    "abc",
    "日本語",
    2.5,
    7,
    True,
    1 << 70,
]
for value in values:
    sys.stdout.write(describe(value) + "\n")


# The same question asked of a statically typed exception: the class is one
# word further in than every other object keeps it, and the taxonomy decides.
def kind(e: Exception) -> str:
    if isinstance(e, ZeroDivisionError):
        return "zde"
    if isinstance(e, ArithmeticError):
        return "arith"
    if isinstance(e, MyErr):
        return "myerr"
    if isinstance(e, ValueError):
        return "value"
    return "other"


excs: "list[Exception]" = [
    ZeroDivisionError("a"),
    ValueError("b"),
    MyErr("c"),
    TypeError("d"),
]
for exc in excs:
    sys.stdout.write(kind(exc) + "\n")


# `bool` under `int`, which is a subclass relation Python has and the ABI does
# not.
def is_int(o: object) -> bool:
    return isinstance(o, int)


sys.stdout.write(str(is_int(True)) + " " + str(is_int(1)) + " " + str(is_int("x")) + "\n")


# A source exception class handed to a parameter declared as its manifest base.
# The base is written by the bare name it was spelled with and has no class of
# its own here, so the only thing that can say `MyErr` is an `Exception` is the
# walk over what the class DECLARED.
def message(e: Exception) -> str:
    return str(e)


sys.stdout.write(message(MyErr("passed")) + " " + message(ValueError("v")) + "\n")


# The proof carried by `and`, which is how `__eq__` is spelled. The right
# operand runs only where the left one holds, so the value it reads is the
# narrowed one -- binding the NAME without the VALUE left the attribute read
# looking at an `object`.
class Point:
    x: int
    y: int

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Point) and other.x == self.x and other.y == self.y


sys.stdout.write(
    str(Point(1, 2) == Point(1, 2))
    + " " + str(Point(1, 2) == Point(1, 3))
    + " " + str(Point(1, 2) == "not a point")
    + "\n"
)
