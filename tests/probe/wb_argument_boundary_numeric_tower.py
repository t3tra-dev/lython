# FIXED 2026-08-15 by SPECIALIZATION, and the prescription this file carried
# for two sessions was the wrong repair. `def f(x: float)` refused `f(3)`; the
# note at TypeSystem.cpp said the fix was "accept the argument here and CONVERT
# it at the call site", and measurement said converting produces wrong answers:
#
#   def p(x: float) -> None: print(x)
#   p(3)            CPython: 3      would-convert: 3.0
#   def q(n: int) -> None: print(n)
#   q(True)         CPython: True   would-convert: 1
#
# CPython does not convert at a parameter boundary. The annotation is inert
# there and the argument keeps its own type, so honouring it by converting
# diverges on every program that can OBSERVE the argument's type.
#
# ⭐ WHAT THE PATHS THAT ALREADY WORKED SAID, which is what located it: an
# inlined method (`C().m(3)` prints 6, `C().m(3.0)` prints 6.0) and a local
# annotated binding (`x: float = 3; print(x)` prints 3) both agree with CPython
# and neither converts. Both are specialization -- one by accident, one by
# construction. A free function was the only spelling that failed, because it
# is the only one whose ABI comes from the annotation rather than from a value.
#
# ⭐ THE MECHANISM. `functionSignature` grew a `monomorphize` mode that re-reads
# a function as if its annotations were absent: parameters take the expected
# callable's types, and the RETURN annotation is ignored so the body walk
# decides the result. That last half is not optional -- `def r(x: float) ->
# float: return x * 2` reached by `r(3)` must return the int 6, and keeping the
# annotation would return 6.0. A lambda has had exactly this for as long as it
# has had parameters, because it has no annotations to read.
#
# Every non-generic top-level function with a plain positional signature is
# registered like a generic (EmitterFunctions.cpp), and a call whose argument
# stands a rung BELOW the declared parameter in the numeric tower gets one body
# per ground signature, memoized on it and capped at 32.
#
# ⭐ THE DECISION IS MADE ON THE EMITTED OPERAND TYPES, not on inference, and
# that is not a refinement -- it is the difference between working and not.
# `types.inferExpr` answers `builtins.float` for `1.0 + 0.0j`, so a
# pre-inference decision specialized `def rotate(z: complex, n: int)` at float
# and emitted a body that could not compile. Caught by
# golden scalar_loop_carried_mutate, which is the only program in the corpus
# that passes a complex literal expression to a complex parameter.
#
# ⭐ A SPECIALIZATION WHOSE BODY CALLS ANOTHER specializable function with the
# narrowed parameter is covered since 2026-10-26 (`outer(4)` below prints 8):
# inference re-reads that call at its argument's rung through the one decision
# the emitter specializes by (TypeSystem::argumentRungSpecialization), so the
# call's type is the value's. The same made a call above the def it reaches
# specialize (the registry is filled when a function is first asked for), and
# kept a decorated def out of it -- its name is the decorator's result.
#
#          def r(x: float) -> float: return x * 2
#          def outer(x: float) -> float: return r(x)
#          print(outer(4))          # CPython 8, and here
#
# ⛔ TWO THINGS IT DOES NOT COVER, both refusals rather than wrong answers:
#
#   1. A body calling ITSELF one rung down: `def power(base: float, n: int)`
#      reached by `power(2, 3)`. Re-reading `power` at int meets `power(base,
#      n - 1)` while the re-reading is in progress, which takes the declared
#      float, and `1` joined with a float is a union the result guard in
#      `argumentRungSpecialization` refuses.
#      Typing it needs a fixpoint over the recursion, not a guess.
#
#   2. A body whose branches return different rungs:
#
#          def pick(n: int) -> int:
#              if n <= 0: return n
#              return 5
#          print(pick(True))        # CPython 5; refused here
#
#      Re-read at bool the result is `bool | 5`, and the annotation is what was
#      collapsing that to int. ⛔ Why NOT collapse it here the same way:
#      CPython does not. `pick(False)` prints False and an int-collapsed result
#      prints 0. The union is the honest type, the py ABI cannot return one,
#      and a refusal is the right outcome -- which is also what this program
#      got before the change, so nothing regressed.
#
# golden: tests/golden/cases/argument_numeric_tower_specialization.py

def f(x: float) -> float:
    return x * 2


print(f(3))
print(f(3.0))
print(f(True))


def p(x: float) -> None:
    print(x)


p(3)
p(3.5)
p(True)


def q(n: int) -> None:
    print(n)


q(True)
q(7)


def mix(a: float, b: int, c: float) -> float:
    return a + b + c


print(mix(1, 2, 3))
print(mix(1.0, 2, 3))
print(mix(1.0, 2, 3.5))
