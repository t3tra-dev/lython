# Helper for an_enum_declared_in_an_imported_module. One of each enum kind,
# because the desugaring differs per base and all three used to die the same
# way when they crossed a file.
from enum import Enum, IntEnum, StrEnum, auto


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class Size(IntEnum):
    SMALL = 1
    LARGE = 2


class Tag(StrEnum):
    A = "a"
    B = "b"


class Step(Enum):
    ONE = auto()
    TWO = auto()
