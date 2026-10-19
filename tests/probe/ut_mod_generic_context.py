# WHAT: A generic class whose __init__ annotates a field with its type
#   parameter, and a base class whose generic method builds one. Run alone
#   it prints nothing.
# KIND: helper module of ut_bug_inherited_generic_method_annotation_in_importer
class Ctx[E: BaseException]:
    def __init__(self, expected: type[E]) -> None:
        self.expected = expected
        self._exception: E | None = None

    def take(self, exc: BaseException) -> bool:
        if isinstance(exc, self.expected):
            self._exception = exc
            return True
        return False


class Case:
    def raises[E: BaseException](self, expected: type[E]) -> Ctx[E]:
        return Ctx[E](expected)
