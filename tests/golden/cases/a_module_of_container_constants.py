# Helper for an_imported_container_constant_is_one_storage. Every container
# spelling a module-level constant takes, plus the functions that read and
# mutate them from inside their own module -- which is the half that used to
# say "unresolved name" at a line where the name is plainly in scope.
# ⭐ AND THE SCALARS WHOSE VALUE IS AN EXPRESSION, which have no literal
# spelling either: the literal channel carries `LIMIT = 3600` because its TYPE
# is the value, and `60 * 60` is a BinOp with nothing to fold into a type.
LIMIT = 60 * 60
BIG = 2**53 + 1
JOINED = "a" + "b"
RATIO = 1.0 / 4.0
FLAG = 1 < 2

NAMES = ["a", "b"]
TABLE = {"k": 1}
PAIR = (1, 2)
TAGS: set[str] = {"x"}
EMPTY: list[int] = []
COUNT = 3


def count() -> int:
    return len(NAMES)


def limit() -> int:
    return LIMIT


def first() -> str:
    return NAMES[0]


def add(name: str) -> int:
    NAMES.append(name)
    return len(NAMES)
