# Helper for an_imported_functions_name. Two plain defs, so the importer can
# reach the same function by three spellings and each must answer with the name
# the `def` gave.
def compute(a: int, b: int) -> int:
    return a + b


def other(n: int) -> int:
    return n
