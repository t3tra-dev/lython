# Helper for a_class_spelling_shadowed_by_an_import: three builtins whose
# spelling is also a CLASS, redefined as plain functions with answers that are
# visibly not the builtin's. Annotations name the builtins: neither CPython
# (lazy annotations) nor this compiler (classes only) reads them as the defs.
def range(n: int) -> list[int]:
    return [n, n + 1]


def str(n: int) -> int:
    return n * 10


def int(s: str) -> str:
    return s + "!"
