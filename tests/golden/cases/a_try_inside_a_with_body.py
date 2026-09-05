# What: a `try` inside a `with` body, with the context manager's target used
# only inside the try. One unwind visits every handler in the chain, so the
# release of the `as` target has to happen once across the whole chain -- on
# the catching arm when something catches, and at the outermost handler when
# nothing does. Runtime values, because both arms have to be taken: a refusal
# is what this shape used to get, and a double free is what the placement it
# was refusing would have run.


class Ledger:
    def __init__(self) -> None:
        self.events: list[str] = []

    def __enter__(self) -> "Ledger":
        self.events.append("enter")
        return self

    def __exit__(self, a: object, b: object, c: object) -> bool:
        self.events.append("exit")
        return False

    def take(self, n: int) -> int:
        if n < 0:
            raise ValueError("negative")
        return n * 2


def caught(n: int) -> int:
    total = 0
    with Ledger() as ledger:
        try:
            total = ledger.take(n)
        except ValueError:
            total = -1
    return total


def propagated(n: int) -> str:
    try:
        with Ledger() as ledger:
            try:
                return str(ledger.take(n))
            except KeyError:
                return "key"
    except ValueError:
        return "value"


def native(path: str) -> bool:
    with open(path) as handle:
        try:
            text = handle.read()
        except OSError:
            text = ""
    return len(text.split("\n")) > 0


def native_finally(path: str) -> bool:
    with open(path) as handle:
        try:
            text = handle.read()
        finally:
            pass
    return len(text) > 0


print(caught(3), caught(-1))
print(propagated(4), propagated(-2))
print(native(__file__), native_finally(__file__))
