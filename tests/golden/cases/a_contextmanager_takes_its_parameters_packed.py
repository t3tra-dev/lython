# WHAT: `@contextmanager` on a method that takes a defaulted `msg` and
# `**params` -- unittest's subTest, which records a failure and lets the loop
# go on -- and on a function that takes `*labels` and `**params`: each `with`
# hands the generator the arguments its call gave, and the generator reads
# them as the tuple and dict they are.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which failures are recorded
# is decided by the assertions at run time. Both were refused ("@contextmanager
# on 'subTest' is not supported with *args or **kwargs"): a generator that
# takes `**kwargs` has no argument lane for the dict.
import contextlib
from contextlib import contextmanager
from typing import Iterator

_sentinel_msg = "<none>"


class TestCase:
    def __init__(self) -> None:
        self.failures: list[str] = []

    @contextmanager
    def subTest(self, msg: str = _sentinel_msg,
                **params: object) -> Iterator[None]:
        try:
            yield
        except AssertionError as e:
            parts = ["%s=%r" % (k, v) for k, v in sorted(params.items())]
            label = ("[" + msg + "] " if msg != _sentinel_msg else "") + \
                "(" + ", ".join(parts) + ")"
            self.failures.append(label + ": " + str(e))

    def test_loop(self) -> None:
        for i in range(4):
            with self.subTest(i=i):
                assert i % 2 == 0, "odd " + str(i)
        for name in ["a", "bb"]:
            with self.subTest("lengths", name=name, n=len(name)):
                assert len(name) == 1, "too long"


t = TestCase()
t.test_loop()
for failure in t.failures:
    print(failure)

failures: list[str] = []


@contextlib.contextmanager
def sub(*labels: str, **params: object) -> Iterator[int]:
    try:
        yield len(labels) + len(params)
    except AssertionError as e:
        failures.append(" ".join(labels) + " " +
                        ", ".join(k + "=" + repr(v) for k, v in params.items())
                        + ": " + str(e))


for i in range(3):
    with sub("loop", "i", i=i, sq=i * i) as n:
        assert i != 1, "i is one (n=%d)" % n
with sub() as n:
    print("no params", n)
print(failures)
