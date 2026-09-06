# Helper for a_builtin_shadowed_by_an_import. Five builtins redefined as plain
# functions, so the importer's calls have an answer that is visibly not the
# builtin's.
def len(v: str) -> int:
    return 99


def abs(v: int) -> int:
    return v + 100


def sum(v: list[int]) -> int:
    return 42


def max(a: int, b: int) -> int:
    return a


def repr(v: int) -> str:
    return "R"
