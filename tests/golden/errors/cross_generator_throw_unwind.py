# Cross-check (p2-genframe x p2-ehfix x p2-traceback): owned str/list live
# across a suspension inside try/finally; throw() unwinds the body, the
# finally runs, the absorbed frame values are released, and the traceback
# matches CPython (throw-site frame + suspended yield frame). The finally
# must not READ the absorbed values: handler-referenced locals crossing a
# yield are still a loud diagnostic (rfc/stdlib-semantics.md, generator
# residual work).
from collections.abc import Generator


def worker() -> Generator[int, None, None]:
    held: list[int] = [1, 2, 3]
    tag: str = "resource-" + "alpha"
    try:
        yield len(held)
        yield held[0] + len(tag)
    finally:
        print("finally ran")


g = worker()
print(next(g))
g.throw(ValueError("injected"))
