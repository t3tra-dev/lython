# WHAT: a context manager's `__exit__(exc_type, exc, tb)` receives the class of
# the exception that left the body (None when none did), and its answer
# decides whether the exception is suppressed -- the shape of unittest's
# assertRaises, written per exception class and as a generic class.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which class `exc_type`
# names, and whether the exception then propagates, is the run time's answer.
# Reading `exc_type` was refused ("an exception's class object ... has no
# value representation") until class values carried their type object.
class AssertRaisesValueError:
    def __init__(self) -> None:
        self.exception: ValueError | None = None

    def __enter__(self) -> "AssertRaisesValueError":
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: object) -> bool:
        if exc_type is None:
            raise AssertionError("ValueError not raised")
        if issubclass(exc_type, ValueError) and isinstance(exc, ValueError):
            self.exception = exc
            return True
        return False


with AssertRaisesValueError() as cm:
    int("x")
print("ok", cm.exception)


class Raises[E: BaseException]:
    def __init__(self, expected: type[E]) -> None:
        self.expected = expected
        self.exception: E | None = None

    def __enter__(self) -> "Raises[E]":
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: object) -> bool:
        if exc_type is None:
            raise AssertionError(self.expected.__name__ + " not raised")
        print("exit with", exc_type.__name__)
        if isinstance(exc, self.expected):
            self.exception = exc
            return True
        return False


with Raises(LookupError) as caught:
    {}["missing"]
print(type(caught.exception).__name__)
try:
    with Raises(KeyError):
        pass
except AssertionError as e:
    print(e)
try:
    with Raises(KeyError):
        raise ValueError("wrong")
except ValueError as e:
    print("propagated", e)
