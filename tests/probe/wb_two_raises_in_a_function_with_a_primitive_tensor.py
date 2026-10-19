# WHAT: `raise cls("first")` and `raise cls("second")` in a function that also
#   computes with a primitive tensor: the second raise lost its message and
#   pushed the first one's traceback frame. The numeric-kernel phase
#   (linalg-lowering) runs upstream passes whose pattern rewriting merges
#   blocks that differ only in their locations, and the two raise blocks are
#   such a pair. A function without a tensor is held out of that phase, which
#   is why the same function without the matrix is right; one that has a
#   tensor still goes through it.
# KIND: バグ (bug) -- silent
# FOUND: 2026-10-20, while making unittest's tracebacks match CPython.
# differential: skip lyrt is Lython's own and CPython cannot import it; the
#   expected output, which is CPython's for the same program on lists, is below
# CLASSIFICATION: silent 誤実行
#   cpython="AssertionError('second')" lyc="AssertionError()"
# Expected: [[2.0, 4.0], [6.0, 8.0]] / AssertionError('first') / [[2.0, 4.0], [6.0, 8.0]] / AssertionError('second')
from lyrt import from_prim
from lyrt.prim import Float, Matrix


def check(cls: type[BaseException], which: bool) -> None:
    m = Matrix[Float[32], 2, 2](
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    print(from_prim(m + m))
    if which:
        raise cls("first")
    raise cls("second")


for which in [True, False]:
    try:
        check(AssertionError, which)
    except AssertionError as e:
        print(repr(e))
