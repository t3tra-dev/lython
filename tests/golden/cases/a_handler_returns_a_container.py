# What: returning an object from inside an `except` handler -- and from the
# `try` body it guards. Running it is what shows the payload survived the
# region: the value has to come back intact from either path, and the one that
# did NOT run must not have left its object behind (the leak gate watches that
# half).
class Record:
    def __init__(self, tag: str) -> None:
        self.tag = tag


def lookup(table: "dict[str, int]", key: str) -> "list[int]":
    try:
        return [table[key]]
    except KeyError:
        return []


def classify(flag: int) -> Record:
    try:
        if flag > 0:
            return Record("positive")
        raise ValueError("not positive")
    except ValueError:
        return Record("other")


def pair(flag: int) -> "tuple[int, str]":
    try:
        if flag > 0:
            return (1, "yes")
        raise ValueError("no")
    except ValueError:
        return (0, "no")


print(lookup({"a": 1}, "a"), lookup({"a": 1}, "z"))
print(classify(1).tag, classify(-1).tag)
print(pair(1), pair(-1))
