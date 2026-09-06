# Helper for an_imported_container_constant_is_one_storage. Every container
# spelling a module-level constant takes, plus the functions that read and
# mutate them from inside their own module -- which is the half that used to
# say "unresolved name" at a line where the name is plainly in scope.
NAMES = ["a", "b"]
TABLE = {"k": 1}
PAIR = (1, 2)
TAGS: set[str] = {"x"}
EMPTY: list[int] = []
COUNT = 3


def count() -> int:
    return len(NAMES)


def first() -> str:
    return NAMES[0]


def add(name: str) -> int:
    NAMES.append(name)
    return len(NAMES)
