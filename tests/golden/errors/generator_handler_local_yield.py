# Owned locals referenced inside a handler region (finally) across a yield
# are still rejected loudly (rfc/stdlib-semantics.md, generator residual
# work: handler-entry block-arg constraint) — never silently mis-executed.
from collections.abc import Generator


def worker() -> Generator[int, None, None]:
    held: list[int] = [1, 2, 3]
    try:
        yield len(held)
    finally:
        print(len(held))


for v in worker():
    print(v)
