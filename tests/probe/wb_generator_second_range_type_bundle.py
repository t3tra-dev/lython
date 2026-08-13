# REFUSED with an INTERNAL message where CPython prints. A generator body
# that builds a second `range` -- nested loops, or one loop after another --
# stops at "new class object has no lowered type bundle", which names a
# lowering invariant and not anything the reader wrote.
#
# MEASURED:
#
#   nested for loops, one yield at the end ........ this file: refused
#   the same two loops in a NON-generator .........  correct
#   one loop before a yield .......................  correct (fixed
#       2026-08-13, tests/golden/cases/generator_loop_before_yield.py)
#   two loops, the yield inside the second ........  refused, same message
#   two sequential loops, a yield in EACH .........  refused, but honestly:
#       "source generator next lowering currently supports only straight-line
#       pure int yield bodies", which is a stated boundary
#
# So this is not the straight-line boundary above -- that one says what it
# does not support. `py.new`'s class object arrives as something that is not
# a TypeObject bundle (lowerNew, Passes/Runtime/Manifest/Calls.cpp), and the
# likely reason is the resume clone's SSA normalization: it threads every
# value that crosses a block boundary through a new block argument
# (GeneratorStateMachine.cpp, `blockValueArguments`), a `!py.type<...>` among
# them, and a block argument has no type-object bundle.
#
# NOT the drain-ordering defect this file was found next to: the refusal is
# identical on the binary from before that repair.
#
# differential: skip the refusal is the recorded state, not a wrong answer
from typing import Iterator


def f(n: int) -> Iterator[int]:
    total = 0
    for i in range(n):
        for j in range(2):
            total = total + i * j
    yield total


for v in f(3):
    print(v)
