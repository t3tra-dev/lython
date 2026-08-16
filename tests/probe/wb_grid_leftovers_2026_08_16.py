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
# (5) FIXED, AND THE PREMISE BELOW WAS WRONG: A SECOND UNION-TYPED READ.
# ============================================================
#     xs = [1, "a"]
#     print(xs[0])
#     print(xs[1])         # runtime manifest has no builtins.list.__getitem__
#
# ⭐ IT WAS NEVER THE READ THAT DEMOTED. Everything from here to the end of
# this section was built on "the first read hands out an alias, so the
# description cannot travel with it", and that rule (`bindRetainedEvidenceBundle`)
# demotes the ELEMENT's contents, not the container's. What demoted the
# container was `demoteCrossBlockContainerEvidence`, which drops the evidence at
# every op outside the block defining the storage -- and printing a union
# branches on the tag, so the second read is in a successor block. Two reads in
# ONE block always worked; that is why `print(xs[0], xs[1])` was fine and
# `print(xs[0]); print(xs[1])` was not, a distinction the reading below never
# accounted for.
#
# The cross-block rule is right for a container something can mutate and vacuous
# for one nothing can. A container whose every use is a read describes the same
# contents in every block, and that is now the exemption
# (`containerContentsAreUnreachableByMutation`). It is a whitelist of uses, not a
# list of mutators, and it reads the same before and after any op is lowered --
# the two properties the cross-block rule was chosen for.
#
# ⛔ Two things had to come with it. Every read must HIT: the evidence tier's
# miss RAISES, spliced into the read's own block, and `i, j = [1]` (where the
# arity check raises first and `[1][1]` is dead) then released a repr twice on
# one path. And the element is in the SLOT's form -- `bool` is stored boxed, its
# ABI is the bare i1, and the union injection counts values rather than checking
# them, so the header went into the i1 lane. bool is the only contract with a
# `box` primitive, which is why every other element type worked.
#
# ⛔ WHAT IS STILL REFUSED: a heterogeneous container MUTATED across a block
# boundary. The exemption declines (correctly -- the contents did change), the
# evidence goes, and the runtime tier still has no `__getitem__` that can
# produce a union. That is the switch the trail below describes, and it is still
# unbuilt. So is iteration: `for x in [1, "a"]` asks for rank-1 memref physical
# values and a union's lane 0 is an i64 tag.
#
# THE TRAIL BELOW IS KEPT AS WRITTEN. It is a correct and expensive
# investigation of a mechanism that was not the cause, and the reason it is
# worth keeping is the second paragraph of it: the ownership kernel really does
# have three guards that assume a single object, and whoever builds the runtime
# switch will meet all three.
#
# ---- the trail, as written before the cause was found ----
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
# ⛔ IT WAS BUILT AND REVERTED, and the measurement is the useful part. A
# `unionElementFromSlotBox` that reads word 1 of the payload box, compares it
# against each member's `runtimeClassIdForContract`, selects the tag and
# selects each member's lanes against a dead value, COMPILES the shapes that
# were refused -- the second read, the module-global cell, `list[int | None]`
# -- and gets one of them SILENTLY WRONG:
#
#     xs = [1, "a", 2.5, None]
#     print(xs[0], xs[1], xs[2], xs[3])   # 1 a 2.5 None   (evidence path)
#     a = xs[0]; b = xs[1]; c = xs[2]; d = xs[3]
#     print(a, b, c, d)                   # 1 1 0.0 None   <- this path
#                                         # CPython: 1 a 2.5 None
#
# ⭐ THE PATTERN NARROWS IT. Every union with ONE non-None member came out
# right (`int | None`, `str | None`), and those are exactly the ones where the
# tag's DEFAULT is already the answer -- the select chain starts at 0 and
# member 0 is the only real member. So the failure is in the SELECTION, not in
# the lane rebuild: with four members, nothing after member 0 was chosen.
#
# ⛔ AND THE SLOT BASE IS NOT IT EITHER. The second attempt dumped the emitted
# IR: the base is `index * 16`, the class-id load is `slot + 1`, and the
# comparison constants are the right ones (`cmpi eq %83, 1` for int, `4` for
# str). Every piece the first two hypotheses blamed is correct.
#
# ⭐ WHAT THE VALUES SAY, three programs, each reading the SECOND element after
# a first read has demoted the evidence:
#
#     xs = ["a", 1]  ; b = xs[1]   ->  1     CORRECT   (int is member 1)
#     xs = [1, "a"]  ; b = xs[1]   ->  ""    wrong     (str is member 1)
#     xs = [1, 2.5]  ; b = xs[1]   ->  0.0   wrong     (float is member 1)
#
# The INT match fires -- member 1 is not the tag's default, so `1` can only
# come from the comparison succeeding. The str and float matches do not: their
# lanes come back as the dead value, which is what an empty str and a 0.0 are.
#
# ⭐ THE THIRD ATTEMPT PRINTED THAT WORD, by boxing it through
# `LyLong_FromI64` and letting the union's int arm print it. It is 4 for a str
# element and 1 for an int element -- the class ids DO agree at run time, so
# the tag comparison was never the problem either.
#
# ⭐ AND A SECOND PROBE ISOLATED THE HALVES: with the tag FORCED to the str
# member and its lanes read from the box words, the read still printed empty.
# So the LANES are what come back wrong, not the tag.
#
# ⛔ WHICH LEAVES ONE PLACE, because everything around it is now eliminated by
# measurement rather than by argument:
#
#   the class ids agree (4 and 1, printed at run time);
#   the slot base is `index * 16` and the class word is `slot + 1` (read off
#     the emitted IR);
#   the lane offsets are `kPointerWordBase + lane` / `kSizeWordBase + lane`,
#     the SAME arithmetic the working homogeneous path uses -- and a
#     homogeneous `["a", "b"]` read through that path after demotion prints
#     `b` correctly;
#   `slotStorageShapesFor` and `unboxSlotElementValues` are identity here --
#     only `builtins.bool` has a `box`/`unbox` primitive, so str, int and
#     float take neither.
#
# ⭐ AND IT IS THE BINDING, found by reading the one place left rather than by
# another build. `retainEvidenceElement` (GetItemOps.cpp) opens with
#
#     if (!ownership::isObjectHeaderLikeType(value.values.front().getType()))
#       return std::nullopt;
#
# and a union's lane 0 is the i64 TAG, not a header. So the retain is skipped,
# `bindRetainedEvidenceValue` binds a plain BORROW of the container's slot, and
# the only thing holding the container past the read is `pinContainerLiveness`
# -- which pins to just after the read, not past the print. The list is freed
# and the str's bytes are read from freed memory, which is why the wrong answer
# is an EMPTY string and a 0.0 rather than garbage: the dead-value shape is
# what a freed payload looks like. An int element survived the same treatment,
# which is why `["a", 1]` printed 1 and looked like a working tag.
#
# ⭐ THAT REPAIR WAS WRITTEN AND THE VALUES ARE NOW RIGHT. Retaining through
# `retainAggregateSlot(op, unionType, values, ...)` -- which consults the tag
# via `forEachActiveUnionMember` -- and binding the bundle with
# `makeObjectBundleWithOwnership(..., Own)` gives CPython's answer everywhere
# the refusal used to be:
#
#     [1, "a"]       read twice -> 1 a / a
#     ["a", 1]                   -> a 1 / 1
#     [1, 2.5]                   -> 1 2.5 / 2.5
#     [1, "a", 2.5, None]        -> all three lines, and both isinstance arms
#
# ⛔ AND IT LEAKS 2 ALLOCATIONS / 81 B, BOUNDED. Measured on
# `tests/leak_gate.py`: the same at 2000 and at 8000 iterations, so it is not
# per-read; zero without a loop; and zero for the identical program over a
# HOMOGENEOUS list, on this binary and on the one before the repair. So the
# leak is the repair's, and its shape is the LAST iteration's owned union
# locals -- three locals (int, str, float) leaking two objects, which is the
# two that are not immortal.
#
# ⭐ THE SUSPECT WAS RIGHT AND THERE IS A THIRD GUARD. `rootOwnedEvidenceElement`
# opens with the same `isObjectHeaderLikeType(values.front())` line, so marking
# the element frame-owned failed for the same reason the retain did, and the
# reference the fix takes was never given back -- that IS the 81 B. Relaxing
# that guard for a union whose lanes include a header gets past it and lands on
# the gate that governs the marker:
#
#     ly.ownership.owned_local_object marks a value this frame never acquired:
#     it is not a fresh allocation, not a call result the contract declares
#     owned, and NO RETAIN ROOTS IT. A value read out of a slot is BORROWED --
#     the slot still holds it -- so the release this token earns would
#     discharge a reference the frame does not have.
#
# Which is correct and is the whole remaining problem stated by the compiler
# itself: `retainAggregateSlot` retained the ACTIVE MEMBER, a lane INSIDE the
# union, and the gate looks for a retain that roots the value the marker names.
#
# ⛔ SO THE MECHANISM IS THREE PLACES IN THE OWNERSHIP KERNEL, not one:
# `retainEvidenceElement`, `rootOwnedEvidenceElement`, and the marker gate,
# each written for a single object and each needing to accept "the retain of a
# union's active member roots the union". That is an extension of the safety
# kernel and belongs in a round of its own -- with the whole trail above as its
# starting point, and with the leak gate as its acceptance test, because the
# two ways of getting it wrong are a use-after-free and a leak and neither
# shows in what a passing suite prints.
#
# The value work is done and reproducible: retain through `retainAggregateSlot`
# plus a bound owned element gives CPython's answer for every program above.
#
# ⛔ A repair here must be red-checked against VALUES, not against compiling.
# The refusal it replaces is loud; the wrong answer it produced is not, and
# every test in the suite passed while it was in place because no golden reads
# a heterogeneous container twice.
#
# ============================================================
# (3) FIXED: `f = lambda v: v * 2` LEARNS FROM THE CALL.
# ============================================================
#     lambda requires a Callable annotation because its type contains
#     unresolved Unknown
#
# The callee-position repair (2026-08-16) reads the parameter types off the
# ARGUMENTS, and an assignment has none. The CALL does, and it is in the same
# suite -- so the assignment scans forward for calls of the name it is binding
# and takes the parameters from them, the way an empty literal takes its
# element type from its seeds. Pinned by
# tests/golden/cases/named_lambda_learns_from_its_calls.py.
#
# ⛔ Two shapes are still refused, and both are the scan DECLINING rather than
# a gap in it:
#
#   x = lambda v: v * 2
#   print(x(1)); print(x("a"))
#     The calls disagree. One body, two parameter types -- emitting it at the
#     first call's types and using it at the second's is a wrong program, not a
#     refused one. CPython accepts this and Lython does not, which is the same
#     line every unannotated generic is on.
#
#   key = lambda p: p[1]
#   print(sorted(pairs, key=key))
#     The name is PASSED, never called. There are no arguments anywhere to read
#     the parameters off; what fixes them is the CALLEE's declared parameter
#     (`sorted`'s `key: Callable[[T], U]`), which is an expectation flowing the
#     other way -- into an argument position, from a manifest contract. The
#     inline form `key=lambda p: p[1]` already works because the expectation
#     reaches the lambda directly. Closing the named form means propagating an
#     argument-position expectation back to the binding, which is a third
#     mechanism.

# ============================================================
# (6) `{1} == frozenset({1})` IS REFUSED, and it is True in CPython.
# ============================================================
#     cannot adapt builtins.frozenset to runtime input 1 of builtins.set.__eq__
#
# Found while folding cross-family `==` (2026-08-16) and deliberately NOT
# folded with it: a set and a frozenset look exactly as different from the
# fold's vantage as a str and an int, and their answer is the opposite one.
# CPython compares them by CONTENTS because both are set objects, so this needs
# `set.__eq__` to accept a frozenset receiver -- a manifest variant, not an
# emitter rule. `[1] == (1,)` is the other direction (False, and a list and a
# tuple really are unequal), so container kinds cannot take one blanket answer
# either way.
#
# ============================================================
# (7) ITERATION over a heterogeneous container is refused.
# ============================================================
#     for x in [1, "a"]:
#     # iteration over a runtime-mode list of '!py.union<int, str>' requires
#     # rank-1 memref physical values, got 'i64'
#
# A union's lane 0 is the i64 TAG, so the iterator's physical-value check
# rejects it. Reading the same elements by index now works, which makes this
# the remaining half of the same shape -- and the same runtime class-id switch
# section (5) describes is what a mutated container's read needs. Iteration
# would additionally need the loop variable to carry the union's lanes across
# the back edge.
#
# ============================================================
# (8) `.get` ON A RECORD: FOUR SHAPES STILL REFUSED.
# ============================================================
# The union-to-union coercion (2026-08-16) closed the common one --
# `doc.get("id")` on `{"id": 1, "name": "x"}` -- and reading the same dict four
# more ways found four separate mechanisms behind it. Each is a REFUSAL; none
# is a wrong answer.
#
#   THE SAME KEY TWICE in one function:
#     doc = {"id": 1, "name": "x"}
#     print(doc.get("id")); print(doc.get("id"))
#     # owned resource from builtin.unrealized_conversion_cast result 0
#     # reaches function exit without release, transfer, or owned return
#   Different keys are fine at any count, so it is the second retain of the
#   SAME object on a second merge edge that has no matching release. Same
#   family as the borrowed-path credit in verifier/runtime/AffineOwnership.cpp:
#   a value lent on two edges is returned once.
#
#   A FLOAT (or any heap-backed) VALUE, with no union in sight:
#     doc = {"s": 2.5}
#     print(doc.get("s"))
#     # ownership: this block-argument merge needs a retain on the edge and
#     # the header prefix cannot be spelled at the point the retain must go
#   ⛔ THE REPAIR WAS BUILT AND REVERTED, and the measurement is the point.
#   The absent arm carries the DEAD placeholder --
#   `memref.get_global @__ly_dead_header_memref_3xi64_`, a CONSTANT whose
#   initializer is the prefix with word 0 = INT64_MAX -- and
#   `prefixIsInitializedAtDefinition` declines it because it is neither a call
#   result nor a block argument. Accepting an immortal constant global (the
#   initializer, not the symbol name: a `dense<0>` global has the same producer
#   and a retain there reads a zero refcount) makes `{"s": 2.5}.get("s")`,
#   `{"s": 2.5, "n": "x"}.get("s")` and a three-member record all print
#   CPython's answer, and the suite stays green except that it moves the
#   SAME-KEY refusal above from "two gets" to "two gets", unchanged. It was
#   reverted because it only converts one refusal into another at a boundary
#   nobody can predict -- 1 get accepted, 2 refused -- and the same-key
#   accounting has to be fixed first for it to mean anything.
#
#   A COMPUTED KEY: "dict __getitem__ evidence candidate 1 has a different
#   physical ABI shape". The dynamic evidence arm selects between candidates
#   with an scf.if chain that yields one shape, and a heterogeneous dict's
#   values do not have one. Each candidate would have to be widened to the
#   union's lanes BEFORE the chain -- the same widening (7) and the mutated
#   container in (5) need.
#
#   AN ABSENT LITERAL KEY (`doc.get("zz")`): the read is statically a miss, and
#   `containerContentsAreUnreachableByMutation` declines a container any of
#   whose reads misses -- because the evidence tier's miss RAISES into the
#   read's own block. Here the raise is dead code under `if k in doc`, which is
#   exactly the `i, j = [1]` shape that rule exists for.
#
import math

# The forms that DO work, so this file runs and the three above stay visible
# as comments rather than as a refusal.
print(math.sqrt(16.0))
print((lambda v: v * 2)(5))
named = lambda v: v * 2
print(named(5))
print(max([("b", 2), ("a", 3)], key=lambda p: p[1]))
mixed = [1, "a", True]
print(mixed[0])
print(mixed[1])
print(mixed[2])
print(mixed[0] == 1, mixed[1] == 1, mixed[2] == True)
if mixed[0]:
    print("truthy")
record = {"id": 1, "name": "x"}
print(record.get("id"), record.get("name"))
a, b = 0, 1
i = 0
while i < 10:
    a, b = b, a + b
    i += 1
print(a, b)
