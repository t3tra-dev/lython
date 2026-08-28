# WHAT: `type(v).__name__` where `v`'s static type says nothing -- an `object`
# element, an `object` parameter, a base-typed receiver whose runtime class is a
# subclass, and a caught exception.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answer is a string the
# program prints, and every way of getting it wrong produces one. A table that
# is missing an id answers with its fallback ("object"), and an exception read
# through the wrong header word answers with the layout it shares
# ("BaseException") -- both read like an answer.
import sys


class Base:
    pass


class Sub(Base):
    pass


class MyErr(ValueError):
    pass


def name_of(o: object) -> str:
    return type(o).__name__


values: "list[object]" = [
    1,
    "s",
    2.5,
    True,
    None,
    [1],
    {"k": 1},
    (1, 2),
    {9},
    b"b",
    2 ** 70,
    Base(),
    Sub(),
    MyErr("m"),
    ZeroDivisionError("z"),
]
for value in values:
    sys.stdout.write(name_of(value) + "\n")

# A base-typed binding whose runtime class is the subclass: this is the case
# the static fold refuses, and the one the class id answers.
b: Base = Sub()
sys.stdout.write(type(b).__name__ + "\n")

# ⛔ AN EXCEPTION KEEPS ITS EXACT CLASS ONE WORD FURTHER IN than every other
# object, and a box copies the word every other object uses.
try:
    raise MyErr("q")
except Exception as e:
    sys.stdout.write(type(e).__name__ + "\n")

erased: "list[object]" = [ValueError("v"), KeyError("k")]
for value in erased:
    sys.stdout.write(name_of(value) + "\n")


# ⛔ A UNION SUBJECT IS BOUND FIRST. The chain that answers one mentions the
# subject once per member, so an EXPRESSION would run N times -- which is why
# this took a NAME only, and why `type(d[k]).__name__` was refused while the
# same program with `v = d[k]` on the line above compiled.
table: "dict[str, int | str]" = {"n": 7, "s": "seven"}
print(type(table["n"]).__name__, type(table["s"]).__name__)

cells: "list[int | str]" = [1, "a"]
print(type(cells[0]).__name__, type(cells[1]).__name__)


def first_kind(items: "list[int | str]") -> str:
    return type(items[0]).__name__


print(first_kind(cells), first_kind(["z", 2]))
