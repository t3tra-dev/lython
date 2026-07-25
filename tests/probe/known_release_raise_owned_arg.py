# probe: REPORTED loud: an owned local is the argument of a raise
# axes: op=raise-arg flow=trybody
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 2

class Err(Exception):
    def __init__(self, xs: list[int]) -> None:
        super().__init__("boom")
        self.xs: list[int] = xs


def run() -> int:
    xs: list[int] = [1, 2]
    raise Err(xs)


try:
    run()
except Err as e:
    print(len(e.xs))
