# WHAT: a name first bound inside a `for` or a `with` that is itself inside a
# `try` is read after the `try`: the last trip's value, the value the `with`
# body bound, and UnboundLocalError when the loop ran no trip.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the value read after the
# `try` is the one the last executed binding left, and whether the loop ran
# at all is a runtime fact.
#
# ⛔ It was "unresolved name": the slot the `try` takes for the name was
# typed by asking whether the name is a TARGET of the statements in it, and a
# `for`'s or a `with`'s body was walked as if it were the target -- and with
# that fixed, the value `x * 2` still could not be typed before the loop had
# bound `x`.


class Held:
    def __init__(self, text: str) -> None:
        self.text = text

    def __enter__(self) -> str:
        return self.text

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc_value: BaseException | None, tb: object) -> None:
        pass


def last_double(xs: list[int]) -> int:
    try:
        for x in xs:
            n = x * 2
    except ValueError:
        return -1
    return n


def inside_with(text: str) -> int:
    try:
        with Held(text) as held:
            size = len(held)
    except OSError:
        return -1
    return size


print(last_double([1, 2, 3]))
print(inside_with("four"))
try:
    print(last_double([]))
except UnboundLocalError as e:
    print("UnboundLocalError:", e)
