# WHAT: A name bound inside a loop body to a value whose release is
#   observable -- an object with `__del__`, a generator suspended in a `try`
#   -- is released at its last use in the iteration, not when the next
#   iteration rebinds the name or the scope ends. CPython prints `made 0`
#   before `del 0`; this prints `del 0` first, because the release runs
#   right after `d.tag` is read and before `print` is called. A generator
#   left at a `yield` inside `try`/`with` is closed (its `finally` runs)
#   before the print that shows what `next` returned.
# KIND: バグ (bug) -- silent, a lifetime shorter than CPython's
# FOUND: 2026-10-20, leak-checking the subTest shape for the unittest port.
#   The same program without the loop (`d = D("0")` ... `d = D("1")`) agrees:
#   the rebinding keep-alive needs the previous value in SSA, and a name that
#   is not read after the loop is not carried around the back edge.
# CPython 3.14 expects:
#   made 0 / del 0 / made 1 / next 1 / finally 0 / next 1 / end / del 1 /
#   finally 1
# differential: expect-wrong a watched name bound in a loop body dies at its last use in the iteration


class D:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __del__(self) -> None:
        print("del", self.tag)


def gen(tag: str):
    try:
        yield 1
    finally:
        print("finally", tag)


for i in range(2):
    d = D(str(i))
    print("made", d.tag)

for i in range(2):
    g = gen(str(i))
    print("next", next(g))
print("end")
