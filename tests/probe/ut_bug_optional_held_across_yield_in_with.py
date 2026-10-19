# WHAT: A generator holds an Optional value (`dict | None`) across a yield
#   that sits inside a `with` block. The ownership verifier stops it: "owned
#   resource ... reaches function exit without release". Without the `with`,
#   or with a non-Optional value, it compiles. unittest's subTest keeps its
#   parent subtest's parameters this way.
# KIND: バグ (bug) -- loud, from the ownership verifier
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   owned resource from @LyDict_FromLength result 0 reaches function exit without release, transfer, or owned return
# CPython 3.14 expects: x 1 / held {'a': 1}
import contextlib


def gen(flag: bool):
    held: dict[str, int] | None = {"a": 1} if flag else None
    with contextlib.nullcontext():
        yield 1
    print("held", held)


for x in gen(True):
    print("x", x)
