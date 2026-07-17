# Cross-check (p2-genframe x p2-traceback): throw() into a generator
# suspended inside an except handler raises at the yield INSIDE that
# handler, so the exception being handled chains under the injected one
# ("During handling of the above exception ...").
from collections.abc import Generator


def gen() -> Generator[int, None, None]:
    try:
        raise ValueError("handler exc")
    except ValueError:
        yield 1


g = gen()
print(next(g))
g.throw(RuntimeError("injected"))
