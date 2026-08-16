# Findings from gridding forty small programs against CPython 3.14 --
# slicing, strings, dicts, comprehensions, builtins, exceptions, closures,
# inheritance, sorting with keys, match, dataclasses, generators, sets,
# unpacking, f-strings, dunders, recursion, list methods, math, Optional. Two
# more from the same grid were repaired (the lambda callee and the loop-carried
# tuple swap); thirty-five agreed outright. Of the three recorded here, the
# first has since been repaired and its measurement is kept because the reason
# it is NOT the Python-parameter rule is the reusable part; two are open.
#
# ============================================================
# (1) FIXED. `math.sqrt(16)` was refused -- an int where a manifest function
#     declares a float parameter.
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
# ⛔ The missing piece was the discriminator, not the coercion. `coerceValue`
# deliberately no longer retypes between the numeric contracts ("that retyping
# was a lie" -- module-global stores report the mismatch instead), so the
# conversion is an emitted `float(x)` and the emitter has to know it is calling
# a manifest export rather than a source function. `freeFunctionContract` is
# that question asked directly: the table holds exactly the manifest's
# `ly.typing.function_contracts`, and a source module's function is reached
# through the same qualified path but is not in it.
#
# golden: tests/golden/cases/manifest_float_parameter_takes_an_int.py
# (red-checked), which keeps `p(3)` printing 3 beside `math.sqrt(16)` printing
# 4.0 -- a repair that converted at both boundaries, or at neither, compiles
# and gets one of them wrong.
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
# (4) AN EMPTY LITERAL SEEDED BY ANOTHER EMPTY LITERAL is still out of reach.
# ============================================================
#     rows = []
#     for i in range(2):
#         inner = []
#         inner.append(i)
#         rows.append(inner)      # 'builtins.object' has no '__getitem__'
#
# `emptyLiteralSeedType` decides `rows` by scanning forward for what seeds it,
# and the seed is `inner`, whose own type is being decided by the same scan one
# level down. The scan pre-binds CONSTANT assignments for exactly this reason
# -- `out = []; k = 0; while ...: out.append(k + 1)` needs `k` -- and a
# container is not a constant. Closing it means iterating the scan to a
# fixpoint, or moving the whole question into the HM engine where an empty
# literal's element is a unification variable that survives to its use.
#
# ============================================================
# (5) A SECOND UNION-TYPED READ of the same container is refused.
# ============================================================
#     xs = [1, "a"]
#     print(xs[0], xs[1])
#     first = xs[0]        # runtime manifest has no builtins.list.__getitem__
#
# The first read demotes the container's contents evidence -- a read hands out
# an ALIAS, so the description cannot travel with it (`bindRetainedEvidenceBundle`
# has the measurement) -- and the later read falls to the runtime path. That
# path returns a boxed object, and nothing widens it into the union: doing so
# means branching on the STORED CLASS ID at run time to pick the tag, which is
# the same switch a boxed union field would need. Reading each element once
# works, which is what `heterogeneous_container_read.py` does.
#
# ⭐ AND THE SAME MISSING SWITCH IS WHY A UNION-ELEMENT CONTAINER GETS NO
# MODULE-GLOBAL CELL. `lowerRuntimeSequenceGetItem` declines a union element
# outright -- `slotStorageShapesFor` puts the tag first and it is an i64, not
# a memref -- so any read that has lost its per-element evidence has nowhere
# to go. A cell hands back the handle and nothing else, which is exactly that
# loss, so `collectModuleGlobals` excludes the shape and it stays value-bound.
# Building the union from the slot's class id closes both at once: the second
# read above, and the cell.
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
