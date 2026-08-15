# Three findings from gridding forty small programs against CPython 3.14 --
# slicing, strings, dicts, comprehensions, builtins, exceptions, closures,
# inheritance, sorting with keys, match, dataclasses, generators, sets,
# unpacking, f-strings, dunders, recursion, list methods, math, Optional. Two
# more from the same grid were repaired (the lambda callee and the loop-carried
# tuple swap); thirty-five agreed outright. These three are what is left.
#
# ============================================================
# (1) `math.sqrt(16)` IS REFUSED. An int where a manifest function declares a
#     float parameter.
# ============================================================
#     static type !py.callable<[!py.contract<"builtins.float">], ...>
#     is not callable: call arguments do not match the Callable contract
#
# ⭐ AND THIS IS NOT THE SAME QUESTION AS `def p(x: float)` REACHED BY `p(3)`,
# which `wb_argument_boundary_numeric_tower.py` settles the other way. There
# the annotation is inert -- CPython leaves the int an int inside the body, so
# the repair was a SECOND BODY at the argument's rung and never a conversion.
# `math.sqrt` is implemented in C against a double: CPython converts through
# `__float__` at the boundary and there is no Python-visible parameter to keep
# an int in. So a manifest float parameter SHOULD coerce where a Python one
# must not, and the two rules are consistent rather than in tension.
#
# ⛔ What is missing is the discriminator, not the coercion. `coerceValue`
# deliberately no longer retypes between the numeric contracts ("that retyping
# was a lie" -- module-global stores report the mismatch instead), so the
# conversion has to be an emitted `float(x)`, and the emitter has to know it is
# calling a manifest export rather than a source function. Every math contract
# is in `ly.typing.function_contracts`; nothing carries that fact to the call
# site today. Same shape for every `float` parameter in the manifest surface.
#
# ============================================================
# (2) `x, y = z = (1, 2)` IS A PARSE ERROR: "expected end of statement".
# ============================================================
# Measured, and the boundary is one pair of parentheses:
#
#     a = b = 3 .................... ok
#     a = b = c = 1 ................ ok
#     z = x, y = (1, 2) ............ ok      (bare tuple target NOT first)
#     (x, y) = z = (1, 2) .......... ok      (parenthesized)
#     x, y = z = (1, 2) ............ PARSE ERROR
#
# So it is a bare tuple target followed by a further target, and only in that
# position. The vendored CPython 3.14 PEG parser accepts the line; the failure
# is in this tree's patches to it or in the AST builder above it
# (`src/lython/parser/CPYTHON_PATCHES.md`). Rare in real code, which is why it
# is recorded rather than repaired -- but a PARSE error on valid Python is a
# different class from a diagnostic, because nothing downstream can report it
# better.
#
# ============================================================
# (3) `f = lambda v: v * 2` STILL NEEDS AN ANNOTATION.
# ============================================================
#     lambda requires a Callable annotation because its type contains
#     unresolved Unknown
#
# The callee-position repair (2026-08-16) reads the parameter types off the
# ARGUMENTS, and an assignment has none. `f: Callable[[int], int] = lambda v:
# v * 2` works, and so does every applied form. Closing this needs the
# assignment to defer the lambda's emission until a call site fixes its
# parameters, which is a different mechanism from an expectation.

import math

# The forms that DO work, so this file runs and the three above stay visible
# as comments rather than as a refusal.
print(math.sqrt(16.0))
print((lambda v: v * 2)(5))
print(max([("b", 2), ("a", 3)], key=lambda p: p[1]))
a, b = 0, 1
i = 0
while i < 10:
    a, b = b, a + b
    i += 1
print(a, b)
