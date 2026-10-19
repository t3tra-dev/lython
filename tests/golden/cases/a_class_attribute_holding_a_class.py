# WHAT: class attributes holding a class value, an `int | None` and a list,
# read through the class, through an instance, and through a base-typed
# value whose subclass redeclares them; the class value is the one an
# assertion raises and tests a raised class against.
#
# WHY THIS IS RUN: which class's attribute a base-typed value reads is its
# runtime class's. A class attribute holding a class could not be read at
# all ("unsupported static class attribute expression"), and a redeclared
# `int | None` or class value was refused through the base.
class Case:
    failureException: type[BaseException] = AssertionError
    maxDiff: int | None = 640
    tags: list[str] = ["base"]

    def fail(self, msg: str) -> None:
        raise self.failureException(msg)


class Mine(Case):
    failureException: type[BaseException] = KeyError
    maxDiff = None
    tags = ["mine"]


class Plain(Case):
    pass


def describe(test: Case) -> str:
    return (test.failureException.__name__ + " " + str(test.maxDiff) + " "
            + str(test.tags))


def is_failure(t: type[BaseException], test: Case) -> bool:
    return issubclass(t, test.failureException)


print(Case.failureException.__name__, Case().failureException.__name__)
print(describe(Case()), "|", describe(Mine()), "|", describe(Plain()))
print(is_failure(AssertionError, Case()), is_failure(KeyError, Case()),
      is_failure(KeyError, Mine()))
for test in [Case(), Mine()]:
    try:
        test.fail("boom")
    except (AssertionError, KeyError) as e:
        print(type(e).__name__, e)
