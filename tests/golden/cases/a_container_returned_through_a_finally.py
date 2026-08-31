# What: returning a container from inside a `finally`-guarded region -- which
# is what every `with` desugars to. Running it shows the finally ran AND the
# payload survived it; the path that did not return yields a default that has
# to be released rather than leaked, which is the half only the leak gate sees.
class Span:
    def __enter__(self) -> "Span":
        return self

    def __exit__(self, kind: object, value: object, tb: object) -> bool:
        print("closed")
        return False


def guarded(table: "dict[str, int]", key: str) -> "list[int]":
    try:
        return [table[key]]
    except KeyError:
        return []
    finally:
        print("checked", key)


def inside_with(values: "list[int]") -> "list[int]":
    with Span():
        return values + [0]


def mapping(flag: int) -> "dict[str, int]":
    try:
        if flag > 0:
            return {"yes": flag}
        raise ValueError("no")
    except ValueError:
        return {}
    finally:
        print("mapped")


class Result:
    def __init__(self, tag: str) -> None:
        self.tag = tag


def held(flag: int) -> Result:
    with Span():
        if flag > 0:
            return Result("kept")
        return Result("other")


print(guarded({"a": 1}, "a"), guarded({"a": 1}, "z"))
print(held(1).tag, held(-1).tag)
print(inside_with([1, 2]))
print(sorted(mapping(3).items()), sorted(mapping(-1).items()))
