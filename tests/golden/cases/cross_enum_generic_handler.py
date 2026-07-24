# Cross-track: wave25/stdlib desugars Enum/IntEnum/StrEnum and NamedTuple at
# compile time, wave25/generics monomorphizes generic functions reached through
# imports (and isolates the imported module's scope), and wave25/defects carries
# an except handler's rebinds out of a try whose body always raises. The three
# meet when a desugared enum member is the value a generic helper returns and
# the value a handler rebinds -- an enum member is a class instance the emitter
# synthesized, so it exercises the same lanes as a user class without being one.
from enum import Enum, IntEnum, StrEnum
from typing import NamedTuple

from generic_import_lib import first, pair_up, count_matches


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class Level(IntEnum):
    LOW = 1
    HIGH = 10


class Mode(StrEnum):
    READ = "r"
    WRITE = "w"


class Point(NamedTuple):
    x: int
    y: int


# An enum member as a dict VALUE and as a dict KEY: members are singletons, so
# identity lookup and hashing both have to reach the same object.
by_name = {"red": Color.RED, "green": Color.GREEN}
print(by_name["red"], by_name["green"])
print(by_name["red"] is Color.RED)
print(by_name["red"].name, by_name["red"].value)

# Built through a helper: a module-level dict global is not visible inside a
# function body (loudly diagnosed, unrelated to this cross).
def make_ranks() -> dict[Color, int]:
    return {Color.RED: 10, Color.GREEN: 20, Color.BLUE: 30}


ranks = make_ranks()
print(ranks[Color.RED], ranks[Color.BLUE])
print(Color.GREEN in ranks)
print(len(ranks))

# Mixin enums as keys alongside their underlying values.
levels = {Level.LOW: "low", Level.HIGH: "high"}
print(levels[Level.LOW], levels[Level.HIGH])
modes = {Mode.READ: 0, Mode.WRITE: 1}
print(modes[Mode.READ], modes[Mode.WRITE])


# A desugared NamedTuple alongside the enums. It stays out of the dicts: two
# int fields already expand past the payload box's handle budget, which the
# container store loudly rejects.
here = Point(5, 6)
print(here, here.x + here.y)
print(here == Point(5, 6), here == Point(1, 2))


# Imported generics instantiated on the enums' underlying types, at two
# different ground types from one registration.
print(first([Level.LOW.value, Level.HIGH.value]))
print(first([Mode.READ.value, Mode.WRITE.value]))
print(pair_up(Level.HIGH.value, Mode.WRITE.value))
print(count_matches([Color.RED.value, Color.GREEN.value, Color.RED.value],
                    Color.RED.value))


# The handler carry: the try body always raises, so the handler's rebinds are
# the only lane out, and every rebound value comes from a desugared enum or
# NamedTuple.
def classify(kind: int) -> str:
    label = "none"
    rank = 0
    try:
        if kind == 1:
            raise ValueError("red")
        raise KeyError("green")
    except ValueError:
        label = Color.RED.name
        rank = Color.RED.value
    except KeyError:
        label = Color.GREEN.name
        rank = make_ranks()[Color.GREEN]
    return label + ":" + str(rank)


print(classify(1))
print(classify(2))


def locate(fail: int) -> str:
    spot = Point(0, 0)
    tag = "start"
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        spot = Point(fail, fail + 1)
        tag = Mode.WRITE.value
    return tag + str(spot.x + spot.y)


print(locate(3))
print(locate(0))


# Enum members carried out of a handler and then used as dict keys after the
# try -- the carried lane must still be the member singleton.
def pick(kind: int) -> int:
    chosen = Color.BLUE
    try:
        raise ValueError("x")
    except ValueError:
        if kind == 1:
            chosen = Color.RED
        else:
            chosen = Color.GREEN
    return make_ranks()[chosen]


print(pick(1))
print(pick(2))


# Iteration order and aliasing survive alongside the imported generics.
for member in Color:
    print(member.name, member.value, ranks[member])
