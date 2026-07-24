from enum import Enum, IntEnum, StrEnum, auto, unique


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    # Equal value: an alias of RED, not a distinct member.
    CRIMSON = 1


class Level(IntEnum):
    LOW = 1
    HIGH = 10


class Mode(StrEnum):
    READ = "r"
    WRITE = auto()


class Counted(Enum):
    FIRST = auto()
    SECOND = auto()
    TENTH = 10
    ELEVENTH = auto()


@unique
class Unique(Enum):
    A = 1
    B = 2


class Loud(Enum):
    ONE = 1

    def __str__(self) -> str:
        return "LOUD"


# str is "Class.MEMBER", repr is "<Class.MEMBER: value>".
print(Color.RED)
print(str(Color.GREEN), repr(Color.BLUE))
print(Color.RED.name, Color.RED.value)

# By-value and by-name lookup return the member singleton.
print(Color(2), Color["BLUE"])
print(Color(1) is Color.RED, Color["GREEN"] is Color.GREEN)

# Aliases share the canonical member's singleton and name.
print(Color.CRIMSON, Color.CRIMSON.name, Color.CRIMSON.value)
print(Color.CRIMSON is Color.RED, Color["CRIMSON"] is Color.RED)

# Equality between members; aliases compare equal to their canonical member.
print(Color.RED == Color.GREEN, Color.RED == Color(1), Color.RED == Color.CRIMSON)

# Iteration skips aliases and follows declaration order.
for member in Color:
    print(member.name, member.value)

# auto() continues from the last explicit value.
for member in Counted:
    print(member.name, member.value)

# IntEnum/StrEnum inherit the mixin's str (the value's own text); repr keeps
# the enum form.
print(Level.LOW, str(Level.HIGH), repr(Level.HIGH), f"{Level.LOW}")
print(Level.HIGH.name, Level.HIGH.value)
print(Mode.READ, str(Mode.WRITE), repr(Mode.READ), f"{Mode.WRITE}")
print(Mode.WRITE.name, Mode.WRITE.value)

# @unique accepts an enum with no duplicate values.
print(Unique.A, Unique.B)

# A user __str__ wins over the synthesized one.
print(Loud.ONE, str(Loud.ONE), repr(Loud.ONE))

# Failed lookups raise (the message text is not compared here: Lython's
# ValueError names the enum but not the offending value).
try:
    print(Color(99))
except ValueError:
    print("ValueError")
try:
    print(Color["MAGENTA"])
except KeyError as error:
    print("KeyError", error)
