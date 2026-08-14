# OPEN, and the record that replaces a WRONG prescription. `def f(x: float)`
# refuses `f(3)`; the note at TypeSystem.cpp used to say the repair was "accept
# the argument here and CONVERT it at the call site", and measurement says
# converting would produce wrong answers rather than fix anything.
#
# MEASURED against python3.14 (./build/bin/lyc, 2026-08-15):
#
#   def p(x: float) -> None: print(x)
#   p(3)            CPython: 3      would-convert: 3.0    <- WRONG
#   def q(n: int) -> None: print(n)
#   q(True)         CPython: True   would-convert: 1      <- WRONG
#
# CPython does not convert at a parameter boundary. The annotation is inert
# there and the argument keeps its own type, so a compiler that honours the
# annotation by converting diverges on every program that can OBSERVE the
# argument's type -- print, str, repr, type-dispatch.
#
# ⭐ AND THE PATHS THAT ALREADY WORK AGREE, which is the strongest evidence:
#
#   C().m(3) / C().m(3.0), m declared (self, x: float) ..... 6 / 6.0   (= CPython)
#   x: float = 3; print(x), in a function ................... 3        (= CPython)
#   x: float = 3; x = 2.5; print both ....................... 3 / 2.5  (= CPython)
#
# Neither converts. The method case is right because the emitter INLINES a
# known method, so the body is emitted against the actual argument type; the
# local case is right because each binding is typed by its own value. Both are
# specialization, reached by accident in one case and by construction in the
# other.
#
# ⭐ SO THE REPAIR IS SPECIALIZATION AT THE FREE-FUNCTION BOUNDARY. A free
# function is not inlined: it is a real func.func whose ABI comes from the
# declared parameter types, which is why this is the only spelling that fails.
# `ensureGenericSpecialization` (EmitterFunctions.cpp:117) already emits a
# second body per instantiation -- memoized on the specialized callable, capped
# at 32 for polymorphic recursion, and memoized BEFORE the body so monomorphic
# recursion resolves to itself. What it keys on is unbound static type
# parameters, so a non-generic `def f(x: float)` never reaches it and its node
# is not retained past emitFunctionDecl.
#
# ⛔ Why NOT widen the acceptance check alone (isSubtypeOfImpl, or the
# bindExpectedType callback in tryCallableApplication): the emitter and lowering
# hold two independent implementations of this relation and they already
# DISAGREE about bool <: int --
#
#   def g(n: int) -> int: return n + 1
#   print(g(True))
#     emit:     accepted
#     lowering: "arguments do not match Callable contract for function target g"
#               (CallablePlanning.cpp:159, via py::isAssignableTo)
#
# -- so widening emit's side alone moves the refusal later, not away, and
# widening both hands the callee an i1 where three int lanes are expected.
# Lowering is the side that is RIGHT here: with no specialization there is no
# body that can receive a bool.
#
# ⛔ Why NOT convert only where the argument is not observable: "observable"
# is not a property of the call site. The callee may print its parameter, and
# whether it does is exactly the whole-program question a per-call rule cannot
# answer.
#
# differential: skip refused; the point is the refusal


def f(x: float) -> float:
    return x * 2


print(f(3))
