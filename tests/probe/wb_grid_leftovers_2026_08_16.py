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
#   TWO LIVE READS OF ONE ELEMENT, one of them a `.get`:
#     doc = {"id": 1, "name": "x"}
#     g = doc.get("id"); v = doc["id"]; print(g, v)
#     # owned resource from builtin.unrealized_conversion_cast result 0 is
#     # still owned when a call to 'LyUnicode_FromBytes' may unwind
#   ⭐ THE GRID NARROWS IT TO ONE SENTENCE, and none of the obvious readings
#   survive it:
#     doc.get("id") twice ................ FAILS
#     doc.get("id") then doc["id"] ....... FAILS
#     doc["id"] then doc.get("id") ....... works   <- ORDER MATTERS
#     doc["id"] twice .................... works   <- the `.get` is required
#     .get("a") and .get("b") ............ works   <- same OBJECT, not same key
#     .get then an UNUSED doc["id"] ...... works
#     .get bound, printed, then read ..... works   <- not both LIVE
#   So it is two live borrows of ONE element where one is a merge-edge lend,
#   and the lend is returned once. Same family as the borrowed-path credit in
#   verifier/runtime/AffineOwnership.cpp (`previousGroups`), which credits a
#   lend returned under a PRE-merge name -- this one is a lend still out when a
#   second name for the same entity is taken.
#
#   ⛔ The order asymmetry is the part a repair has to explain. A retain
#   inserted for the merge DOMINATES the later read's consume point and is
#   counted by `aggregateRetainsHeldAt`; inserted after, it is not. Any repair
#   that only adds a release will pass this program and change the other order.
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
# ============================================================
# (9) THE GENERATOR UNPACK BINDING MOVED FOUR REFUSALS ONE LAYER DEEPER.
# ============================================================
# `a, b = 0, 1` in a generator body now types its yield (2026-08-16); the four
# shapes below were failing BEFORE that change and still fail, each with a
# different message than it had. None became a wrong answer, and the new
# messages are the useful part -- they name the next mechanism instead of the
# yield type:
#
#   def pairs() -> Iterator[tuple[int, str]]:
#       for i, s in [(1, "a"), (2, "b")]: yield i, s
#   was "generator function is annotated Iterator[...] but yields ...";
#   now "source generator next lowering currently supports only straight-line"
#   -- the recorded for-in-generator limit, which is what it really needs.
#
#   for k, v in sorted(d.items()): yield k    -- same limit, unchanged message.
#
#   for a, b in zip(xs, ys): yield b * a
#   was that limit; now "static type builtins.object is not callable". The
#   lenient walk cannot infer `zip(...)`, so the tuple distribution has nothing
#   to distribute. Typing the lazy-iterator builtins in that walk is a
#   contained follow-up.
#
#   (a, b), c = (1, 2), 3      -- NESTED unpack
#   a, b = xs                  -- unpack from a LIST
#   were both "runtime bundle for builtins.object has 5 values"; now
#   "owned resource from @LyTuple_FromLength / @LyList_FromLength result 0
#   reaches function exit without release". The names are typed; what is left
#   is the temporary the unpack builds inside a generator frame, which nothing
#   releases across a suspension.
#
# ============================================================
# (10) `print(x, end="")` IS REFUSED, and the SINK is why (sep= is fixed).
# ============================================================
#     static type !py.callable<[], vararg = tuple[object], returns = [None]>
#     is not callable: call arguments do not match the Callable contract
#
# `builtins.print`'s contract has no keyword parameters, and `tryEmitPrintCall`
# declined the moment it saw one -- so the report named neither the keyword nor
# the reason. `sep=` was free (the emitter already builds the space-joined
# string; a different separator is a different constant) and is now supported.
#
# ⛔ `end=` is not, and the blocker is the SINK rather than the join. There is
# exactly one builtin write, `LyUnicode_PrintLine`
# (`ly.runtime.builtin = "print"`, `builtin_lowering = "method_sink"`), and it
# appends the newline. `LyUnicode_Print` next to it does not, and nothing names
# it as a builtin, so the emitter cannot reach it. Closing this means a second
# builtin sink -- a manifest entry, a TypeSystem binding and a synthesized
# callee, which puts a pseudo-builtin name in the user's namespace -- or
# routing through `sys.stdout.write`, which the emitter would have to reach
# without an import. Both are design calls, not wiring. Until then the refusal
# says which keyword and why (EmitterTests.NamesThePrintKeywordItCannotTake).
#
# Grouped with the missing modules rather than with the defects: `re`,
# `collections.deque` / `namedtuple` / `defaultdict` and
# `itertools.islice`-as-a-value are refused the same way and for the same
# reason -- nobody wrote them yet.
#
# ============================================================
# (11) A BORROWED UNION PARAMETER RETURNED AS AN OWNED UNION RESULT LEAKS.
# ============================================================
#     def f(k: str, d: str | None = None) -> str | None:
#         if k == "a":
#             return k + "-x"
#         return d              # <- the borrowed parameter
#     print(f("a"))             # leaks 43 B, one allocation
#
# And the same value used twice is a REFUSAL rather than a leak:
#     v = f("a"); print(v); if v: ...
#     # released owned resource from @f is used after release (by call to
#     # 'LyUnicode_Bool')
#
# ⭐ FOUR MEASUREMENTS BOUND IT, and each kills a wider reading:
#     return None instead of `return d` ............ clean  (the PARAMETER arm)
#     `-> str` with `d: str = "z"` ................. clean  (the UNION)
#     `list[int] | None` with the same shape ....... clean  (the STR member)
#     `if d is None: return None` / `return d` ..... clean  (the NARROWING)
#
# So it is one shape: a union-typed parameter forwarded WHOLE as a union-typed
# result, where a member owns a reference. Returning the narrowed member
# instead re-wraps it and the ownership is consistent again, which is why
# `os.getenv` is written with the two-line arm and a ⛔ note pointing here.
# The caller's contract says the result is owned; the borrowed-parameter path
# never took a reference for it to release.
#
# ⛔ This is what blocked `os.getenv` returning None, and os.py's docstring had
# the reason wrong: it said an `Optional[str]` return "has no physical layout
# across the native boundary yet". The layout is fine and the native call
# still returns a str -- the None never crosses the boundary. What was missing
# was this.
#
# ============================================================
# (12) FIXED: `.items()` ON A COMPREHENSION RESULT, USED DIRECTLY.
# ============================================================
#     xs = [1, 2]
#     print(sorted({x: x * x for x in xs}.items()))
#     # runtime manifest has no builtins.dict.items method
#
# ⛔ THE FIRST READING WAS WRONG and is kept here because it looked right: "the
# comprehension's result arrives without the mapping EVIDENCE the method
# needs". It is not evidence. `keys`/`values`/`items` have no runtime object at
# all -- they are sugar the emitter answers by iterating the dict -- and the
# sugar asks `isDictTypedExpr(receiver)`. The type walk had NO ARM for a
# comprehension, so every question about one used directly answered
# `builtins.object`.
#
# The three measurements that should have pointed there: binding it to a name
# first works (the SYMBOL carries the type), a dict LITERAL temporary works
# (the literal has an arm), and `len()` works (its own sugar rewrites
# `len(d.items())` to `len(d)` before asking). All three are about the TYPE,
# and "evidence" was a guess that fit two of them.
#
# ============================================================
# (13) `type(x)` IS UNBOUND, and implementing it statically would be wrong.
# ============================================================
#     print(type(e).__name__)          # unresolved name 'type'
#     print(e.__class__.__name__)      # attr.get object type has no class schema
#
# `type(x)` for a statically known, unsubclassed x IS computable here --
# `py.type.object` exists and `type[X]` carries no value -- but the commonest
# use is the one that would be WRONG:
#
#     except Exception as e: print(type(e).__name__)
#
# CPython answers with the RUNTIME class (ValueError); the static type is the
# caught one (Exception). Answering statically would print "Exception" and
# never say so. Doing it properly means reading the class id out of the header
# and mapping it to a name at run time, which is a runtime table this tree does
# not expose. Left unbound deliberately, with this note as the reason.
#
# ============================================================
# (14) `isinstance(o, C)` ON AN `object` PARAMETER: the test works, the
#      NARROWED VALUE does not.
# ============================================================
#     def __eq__(self, o: object) -> bool:
#         if isinstance(o, Money): return self.cents == o.cents
#         return False
#     # isinstance on an object-typed value requires dynamic object
#     # inspection, which is excluded from the static evidence kernel
#
# That is the shape typeshed declares for `__eq__` and every hand-written
# comparison copies, so a class cannot implement equality the way Python is
# written. The refusal reads as policy and is guarding something narrower.
#
# ⛔ IT WAS BUILT AND REVERTED, and the measurement is the point. `py.class.test`
# is a CLASS-ID comparison -- the one a base-typed receiver tested for a
# subclass already uses -- and an erased `object` is header-fronted for the
# reason `object.__str__` can dispatch on it. Letting an object top reach that
# arm (gated on a source-class target, since a manifest contract has no
# `py.class` schema) makes the TEST right everywhere:
#
#     class C:
#         def __init__(self, a: int) -> None: self.a = a
#         def peek(self, o: object) -> int:
#             if isinstance(o, C): return o.a
#             return -1
#     print(C(1).peek(C(7)))          # 7   CORRECT
#
#     def f(o: object) -> int:
#         if isinstance(o, C): return o.a
#         return -1
#     print(f(C(1)))                  # 40622248832   <- a POINTER
#
# ⭐ METHOD vs FREE FUNCTION is the whole difference, and both were measured on
# the same class. Inside a method the receiver's class is in hand and the
# narrowed value gets the layout; in a free function the erased object has none
# to recover, and the field read walks the wrong words. A silent wrong answer,
# so the change was reverted whole.
#
# What it needs: the narrowing must RE-MATERIALIZE the value from the erased
# payload (the unbox the diagnostic calls "dynamic object inspection"), not
# merely re-type it. The class test itself is ready.
#
# ⛔ And a BUILTIN target is a second, smaller gap even then: `isinstance(o,
# int)` has no `py.class` schema to match against, so it would still refuse.
#
# ============================================================
# (15) `set.update` TAKES A SET, NOT AN ITERABLE.
# ============================================================
#     s = {1}
#     s.update([2, 3])
#     # static type !py.contract<"builtins.set", [...]> does not provide
#     # manifest method 'update'
#
# ⛔ CORRECTING THIS NOTE, which first said the method was "not declared". It
# is: `builtins.mlir`'s set contract has `update` at index 9, declared
# `Callable[[set, set[$T]], None]`, and `LySet_UpdateM` takes a second
# `memref<11xi64>` -- another SET. `s.update({2, 3})` works, and so do `|=`,
# `difference_update` and the rest. What is missing is CPython's "any iterable"
# argument, which needs either a native that walks a list payload (the element
# ABI varies, which is why it does not exist) or an emitter desugaring into a
# loop of `add`. The diagnostic names the receiver, not the argument, which is
# what made the first reading look right.
#
# ============================================================
# (16) A NAME BOUND INSIDE A `try` DOES NOT REACH ITS `else`.
# ============================================================
#     try:
#         n = v
#     except ValueError:
#         return "raised"
#     else:
#         if n > 0:          # unresolved name 'n'
#
# CPython runs the else in the enclosing scope, so `n` is bound there. Here a
# binding CREATED inside a try body does not escape it -- the same rule the
# `next(it, default)` desugar works around by pre-binding its result name, and
# the same one that forced the source-iterator loop rewrite to put its body in
# the try's ELSE rather than after it. Rebinding a PRE-EXISTING local works, so
# `n = 0` before the try is the workaround, and it is what
# tests/golden/cases/try_else_with_a_nested_statement.py does.
#
# Found 2026-08-17 while writing that golden, which pins a different defect (an
# `if` in the else clause was "empty block: expect at least a terminator").
#
# ============================================================
# (17) `"-".join(w)` WHERE `w` IS AN ITERABLE SOURCE CLASS.
# ============================================================
#     static type !py.contract<"builtins.str"> does not provide manifest
#     method 'join'
#
# CPython's join takes any iterable; no manifest parameter declares a user
# class, so the inference for the ARGUMENT fails and the message blames the
# receiver. `"-".join(list(w))` works, and a GENERATOR argument is materialized
# for exactly this reason (2026-08-17).
#
# ⛔ EXTENDING THAT REWRITE TO SOURCE CLASSES WAS TRIED AND REVERTED: gating on
# "has `__iter__`, or `__len__` and `__getitem__`" took out 8 goldens
# (json_value_repr, stdlib_json_accessors, stdlib_json_build and five more).
# `json.JSONValue` satisfies the predicate and is passed to manifest methods
# that want the OBJECT, not a list of its elements -- so the predicate has to be
# the PARAMETER's declared type rather than the argument's shape, and that path
# infers the method's contract from the actual argument types after emitting
# them. Same structural gap as the manifest-parameter expectation.
#
# ============================================================
# (18) THREE SHAPES THE `float | int` UNION STILL REFUSES.
# ============================================================
# `1.5 if c else 0` types as `float | int`, which is RIGHT -- CPython's false
# arm is the int 0. The merge itself was fixed 2026-08-17 (the immortal constant
# global the converted literal lives in is now accepted as prefix-initialized,
# tests/golden/cases/float_merge_needs_a_retain.py). What the union cannot do
# yet:
#
#   bound = 1.5 if c else 0
#   bound + 1
#   # union<float, int> does not provide manifest method '__add__'
#     -- the per-member dispatch `==` got, for the arithmetic operators. The
#     result types differ per member (float + int is float, int + int is int),
#     so the arm has to join them the way the tower does.
#
#   values: list[float] = [1.5 if c else 0]
#   # runtime object header has invalid type 'i64'
#     -- the tower conversion at a CONTAINER boundary. The element is declared
#     float and the union arrives whole.
#
#   def f(c: bool) -> float: return 1.5 if c else 0
#   # cannot adapt  return value to callable return ABI 0
#     -- the same conversion at the RETURN boundary.
#
# All three are the union meeting a declared float, which is one question asked
# in three places.
#
# ============================================================
# (57) FIXED: max(xs, default=...) / min(xs, default=...).
# ============================================================
#     print(max(xs, default=0))
#     # max() with the 'default' keyword argument is not supported
#
# ⭐ FIXED 2026-08-20. The fold already emits a seen-flag and `if not seen:
# raise ValueError(...)`, so a default is that branch's other answer -- and the
# cheapest way to give it is not a second arm but a different SEED. The
# accumulator starts at the default and the empty guard disappears.
#
# ⛔ SEEDING THE PLACEHOLDER AND ASSIGNING THE DEFAULT AFTERWARDS COMPILED AND
# LEAKED: "owned resource from builtin.unrealized_conversion_cast result 0
# reaches function exit without release". The fabricated seed was only ever
# unread because the empty path RAISED; give that path a return and the
# fabrication reaches the exit. The measurement is the argument for seeding
# rather than branching, and it is why the empty guard is now absent rather
# than two-armed.
#
# ⛔ AND THE KEYWORDS ARE NOW READ BY NAME. The old code took
# `reducerKeywords->front()` as "the key", which was right while `key` was the
# only keyword allowed; `min(xs, default="none", key=len)` then probed the
# DEFAULT as if it were the key function and reported "needs a key the fold can
# seed ... this one produces object". Both orders are in the golden.
#
# Pinned by tests/golden/cases/min_max_with_a_default.py.
#
# ============================================================
# (56) FIXED: setattr(x, "v", value) -- AND WHAT THE WHOLE BUILTIN SURFACE
#      STILL LACKS, MEASURED RATHER THAN GUESSED.
# ============================================================
#     setattr(c, "v", 5)       # unresolved name 'setattr'
#
# ⭐ FIXED 2026-08-19. Written as a call, it is the store: with a literal name
# `setattr(x, "v", v)` IS `x.v = v`, so it becomes one and the field's declared
# type, the release of the value it replaces and every refusal come from the
# assignment path rather than from a second implementation of it. The call's own
# value is None, which is what CPython's returns -- `print(setattr(...))` prints
# None in both.
#
# ⛔ A NAME BOUND TO ONE LITERAL STILL WORKS, because the inference has already
# folded it: `name = "v"; setattr(c, name, 2)` compiles. What is refused is a
# name with no static answer (a str PARAMETER), where there would be no field
# for the store to land on. Both halves are pinned in EmitterTests, since a
# refusal is not worth an execution.
#
# ⭐ AND THE SURFACE WAS MEASURED, not sampled: a generated probe bound every
# name in CPython's builtins and asked which this compiler rejects. Exactly
# three were unbound -- `bytearray`, `id`, `setattr` -- and setattr was the one
# that is a rewrite of something already here. `id` needs an identity notion
# this compiler has not committed to for unboxed values, and `bytearray` is a
# mutable type, not a missing name. So "which builtins are missing" is now a
# closed question with two entries, both of which need something built.
#
# Pinned by tests/golden/cases/reflection_builtins.py, alongside the three names
# from (54) -- the setattr half loops a str field so the replaced value's
# release is exercised, which an int field would not show.
#
# ============================================================
# (55) FIXED: zip(a, b, strict=True).
# ============================================================
#     print(list(zip([1, 2], "ab", strict=False)))
#     # zip() takes no keyword arguments
#
# ⭐ FIXED 2026-08-19 (evening). Every keyword was refused, which refused the one
# Python 3.10 added and tells readers to prefer. False is what zip already does;
# True adds the length check, and CPython names which argument differs and in
# which direction -- so the check is per argument, in argument order, and the
# first mismatch is the one reported. Both messages are matched exactly.
#
# ⛔ The flag must be a LITERAL: the two answers are different emitted code, not
# a different value.
#
# ⛔ strict=True needs the FIRST argument to be indexable too, which plain zip
# does not require of it (it drives the loop with that one and indexes the
# rest). A leading iterator has no length to compare, and the refusal says so
# rather than comparing something else.
#
# Pinned by tests/golden/cases/zip_strict.py.
#
# ============================================================
# (54) FIXED: hasattr / getattr / callable.
# ============================================================
#     print(hasattr(x, "v"))     # unresolved name 'hasattr'
#     print(getattr(x, "v"))     # unresolved name 'getattr'
#     print(callable(f))         # unresolved name 'callable'
#
# ⭐ FIXED 2026-08-19 (evening). All three are compile-time questions here: the
# attribute either exists on the static class or it does not, and a value either
# has a callable contract or it does not. `getattr` with a literal name IS the
# attribute lookup written as a call, so it is rewritten to one and inherits
# every attribute rule.
#
# ⛔ A SUBCLASS CAN ONLY ADD, which makes the answers asymmetric: True stands
# (the base has it, so every instance does), and False is REFUSED when the class
# has a subclass, because the subclass may define exactly that name. Answering
# False there is the silent wrong answer; the refusal names the subclass.
#
# ⛔ Refused for want of anything static: a computed attribute name, and
# `getattr(x, "v", default)` (its arm choice would need the hasattr fold).
#
# ⛔ `id(x)` is still unbound. It is the object's address, which the default repr
# already formats, so it wants a manifest primitive returning an int from a
# header rather than a fold.
#
# Pinned by tests/golden/cases/reflection_builtins.py.
#
# ============================================================
# (53) FIXED: THE DEFAULT REPR NAMED THE CLASS THE VALUE WAS HELD AS.
# ============================================================
#     class A: pass
#     class B(A): pass
#     x: A = B()
#     print(x)      # <__main__.A object at 0x...>
#                   # CPython: <__main__.B object at 0x...>
#
# ⭐ FIXED 2026-08-19 (evening), on the class-name table built for
# type(v).__name__ an hour earlier. The prefix was baked in at compile time from
# the STATIC contract, so the value reported the class it was HELD as. Nothing
# could see it: the address differs between runs so no output comparison reads
# that far, and there was no diagnostic -- the compiler was sure. THIS IS THE
# SILENT BUCKET, found by writing `print(x)` for a base-typed variable while
# looking at something else.
#
# ⛔ Manifest objects keep the compile-time prefix: their header word 1 is not a
# class id, and the contracts that reach the default repr have no subclass to be
# wrong about.
#
# ⭐ AND THE ERASED READER, immediately after: an instance passed through an
# `object` parameter (or held in a list[object]) is boxed and rendered by the
# manifest dispatch, which reads the BOX -- and the box carries the class id in
# the same word, so it names the real class too. An int or a str going the same
# way still prints its own repr, because the hook answers first and only the
# fallback reaches this.
#
# Pinned by tests/golden/cases/default_repr_names_the_real_class.py.
#
# ============================================================
# (52) FIXED: type(x), AND IDENTITY BETWEEN TYPE OBJECTS.
# ============================================================
#     print(type(x).__name__)     # unresolved name 'type'
#     print(type(x) is C)         # `is` requires reference-typed operands
#
# ⭐ FIXED 2026-08-19. The name was unbound, which took the standard "what did I
# get" idiom with it. `type(x)` is answerable statically exactly when nothing can
# put a SUBCLASS instance in x -- a manifest contract is its own runtime class
# here (a bool is a truth bit, not an int), and a source class is too unless the
# program declares a subclass of it. The subclass scan over classMros IS the
# soundness of the fold, not a nicety: `x: A = B()` makes the static class A and
# the runtime class B.
#
# ⛔ NOT bound as the `type` CLASS: that would make `type(x)` an instantiation,
# and a type object built from an instance is not what CPython returns.
#
# ⛔ The argument still runs -- `type(f())` calls f, once, which the golden's
# counter is there to show.
#
# ⭐ AND `is` BETWEEN TWO TYPE OBJECTS folds: a class has exactly one type object
# in CPython, so the answer is whether they name the same class. `C is C` was
# refused too, which is how the exact-class test lost both of its spellings.
#
# ⭐ AND THE EXCEPTION CASE, built the same day rather than left. The fold cannot
# answer it -- a handler's static class is the one CAUGHT and CPython prints the
# one RAISED -- but an exception instance carries its dynamic class id in its
# header, which is what the traceback and the repr already read. So
# `type(e).__name__` lowers to a read of that id: a BaseException
# `__class_name__` manifest method (off the typed surface, like str.__int__) and
# a `py.class_name` op beside repr/str/int/float, which is the shape this dialect
# already uses for "dispatch one named method". A SOURCE exception class has no
# manifest method of its own, so the receiver is retyped to its exception
# ancestor first -- the same retyping the print path does, and what keeps a user
# class answering its own name (NotFound, not AppError).
#
# ⭐ AND THE POLYMORPHIC NAME, 2026-08-19 (evening): a value whose static class
# has subclasses now answers `type(v).__name__` too, from the class id its header
# carries in word 1 -- the word isinstance already reads -- through a per-program
# class-name table the lowering synthesizes beside the user-exception one. So
# `type(shape).__name__` over a list of Shape prints Circle, Square, Shape.
#
# ⛔ The table's generator must REPLACE the manifest's external declaration of
# the symbol rather than treat it as "already generated": builtins.mlir declares
# __ly_source_class_name because it calls it, so the naive "if the symbol exists,
# return" left nothing defined and the JIT said "Symbols not found".
#
# ⛔ A manifest EXCEPTION contract keeps the manifest __class_name__ path. Its
# header carries the class id in a different word than a source instance's, and
# routing it through the generic reader printed "object" for a caught
# ValueError.
#
# ⭐ AND A UNION ANSWERS BY TAG: `type(v).__name__` over `int | str | float` is
# the member's name chosen by the same test isinstance uses, written as the
# conditional expression a reader would have written -- so the tag tests, the
# narrowing and the string constants all come from paths that already exist. The
# subject must be a NAME (the chain mentions it once per member), and every
# member must be exact, which is the same subclass question asked per member.
#
# ⭐ AND THE TWO OTHER SPELLINGS OF THE SAME QUESTION: `x.__class__` (which
# reached the FIELD lookup and reported "class C has no field '__class__'") is
# rewritten to `type(x)` and takes the same road, dynamic read included; and
# `type(a) == type(b)` folds like the `is` spelling, because a class has exactly
# one type object and a reader picks either. `==` had reached the manifest
# dispatch: "!py.type<...> does not provide manifest method '__eq__'".
#
# ⛔ `type(v) is B` for a subclassed static class stays refused, and correctly:
# the type OBJECT would have to be a runtime value, which is a different
# mechanism from the name. `print(C)` (a type object as a printed value) is
# still refused too: "runtime method receiver has no concrete contract".
#
# Pinned by tests/golden/cases/type_of_a_value.py.
#
# ============================================================
# (51) FIXED: import os.path.
# ============================================================
#     import os.path
#     print(os.path.join("a", "b"))    # unsupported import 'os.path'
#
# ⭐ FIXED 2026-08-19, and the cause was in the DRIVER, not in the emitter where
# every earlier attempt went. The driver decides which stdlib sources to compile
# from the import statements, and for a dotted name it requested only prefixes
# that are PACKAGE DIRECTORIES. `os` is a module (os.py, which does `import
# posixpath as path`), so `import os.path` requested nothing for it, os.py was
# never compiled, and the emitter's `lookupSourceModule("os")` was null -- which
# is exactly what the old note observed without asking why. Requesting every
# prefix that resolves to a source at all is the fix; the emitter then binds the
# root the way `import os` always did.
#
# ⛔ The old note's list of failed repairs is worth keeping in mind as a shape:
# four of them were binding a module that did not exist yet. When a binding
# "does nothing at all", ask who was supposed to create the thing being bound.
#
# ⭐ AND THE OTHER TWO SPELLINGS, once the first was understood. `import os.path
# as p` and `from os.path import join` bind the SUBMODULE rather than the root,
# and the submodule is a name inside the root's own body (`import posixpath as
# path`) -- so asking the root what it publishes under that name turns both into
# a namespace binding the emitter already had. The from-import needed the same
# prefix request in the driver: asking only for "os.path", which names no source,
# asked for nothing at all.
#
# Pinned by tests/golden/cases/dotted_import_binds_the_root.py.
#
# ============================================================
# (50) FIXED: int(s, base=16) -- THE KEYWORD SPELLING.
# ============================================================
#     print(int("ff", base=16))
#     # static type !py.contract<"builtins.int"> does not provide manifest
#     # method '__init__'
#
# ⭐ FIXED 2026-08-19, found by asking which OTHER keyword calls the manifest
# keyword path (49b) had unlocked. This one is not a manifest call at all: the
# two-argument int() is an emitter interception that synthesises a helper, and it
# read the base POSITIONALLY -- it declined the moment a keyword appeared, so the
# call fell through to class instantiation and the diagnostic talked about int's
# missing __init__ rather than about the argument.
#
# Also checked in the same sweep and already working: sorted(reverse=True),
# xs.sort(reverse=True), round(x, ndigits=2). Still refused with its own
# documented diagnostic: print(sep=..., end=...).
#
# Pinned by tests/golden/cases/int_base_keyword.py, whose bases are 2 and 16 so
# the digit set is consulted, and whose third call keeps the whitespace-and-
# prefix form -- forwarding the keyword to the wrong parameter would still print
# a number for "ff".
#
# ============================================================
# (49) FIXED: THE COUNTING IDIOM SEEDS ITS OWN DICT.
# ============================================================
#     counts = {}
#     for w in words:
#         counts[w] = counts.get(w, 0) + 1
#     # !py.union<int, object> does not provide manifest method '__add__'
#
# ⭐ FIXED 2026-08-19. An empty literal has no element type, so the emitter scans
# the suite for a store that says what goes in -- and that scan SKIPS any stored
# expression mentioning the name, because reading `counts` while deciding what
# `counts` holds reads it at the type being decided. In this idiom the
# self-referential store is the only one there is, so the dict stayed at object.
#
# The `.get(key, default)` inside it carries the answer: the default is the value
# when the key is absent. Bind that provisionally, re-infer the WHOLE stored
# expression, and `+ 1` seeds int while `+ 1.5` seeds float -- taking the
# default's type directly would have got the second one wrong.
#
# ⛔ Inside the walk, not after it: the stored expression mentions the LOOP
# TARGET, which the walk binds in a scope that is gone by the time it returns.
#
# ⛔ A fallback, not a preference. A store that does not mention the name is
# better evidence and still wins, and two disagreeing stores are still a
# disagreement.
#
# ⛔ `counts.get(w, 0) + 1.5` over a dict of FLOATS stays refused, and did before
# this too (an annotated dict[str, float] fails identically): get's third overload
# answers `V | D`, and `float | int` has no __add__. `0.0` as the default is the
# fix and is what CPython's typing says as well.
#
# Pinned by tests/golden/cases/counting_idiom_seeds_the_dict.py.
#
# ============================================================
# (48) FIXED: str(b, "utf-8") IS b.decode("utf-8").
# ============================================================
#     print(str(b"ab", "utf-8"))
#     # static type !py.contract<"builtins.str"> does not provide manifest
#     # method '__init__'
#
# ⭐ FIXED 2026-08-19. The diagnostic named str, and str was not what was
# unsupported: a second argument makes the call a DECODE, and the runtime has had
# decode in all three arities (no-arg, encoding, encoding+errors) all along. The
# one-argument spelling stays the __str__ dispatch, which is why `str(b"ab")`
# was already right (it prints the repr, b'ab') while this form was not.
#
# ⛔ Positional only. A keyword spelling (`str(b, encoding="utf-8")`) still falls
# through to the class path; CPython accepts it, so that is a remaining gap
# rather than a decision.
#
# ⛔ `str(5, "utf-8")` is refused statically where CPython raises TypeError at
# run time -- the project's rule, since the argument type is known at the call.
#
# Pinned by tests/golden/cases/str_of_bytes_with_encoding.py, whose "hé" is two
# bytes in and one character out: a decode that is a memcpy prints the same
# thing for pure ASCII.
#
# ============================================================
# (47) FIXED: complex(...) -- THE NAME AND THE CONSTRUCTORS.
# ============================================================
#     print(complex(1, 2))     # unresolved name 'complex'
#     print(1 + 2j)            # (1+2j) -- the SAME type, one line up
#
# ⭐ FIXED 2026-08-19. The type was fully there: the manifest carries add / sub /
# mul / truediv / neg / pos / abs / eq / ne / repr / str and a __new__ taking two
# f64 with defaults, and literals reach all of it. What was missing was the name
# binding plus the class's own __new__ / __init__ declarations -- so the spelling
# a CPython user reaches for first was the only one that did not work.
#
# ⛔ The empty __init__ is load-bearing. __new__ builds the whole value, but the
# constructor path emits py.new AND py.init, and with no __init__ of its own the
# MRO's next provider is builtins.object's, whose input is a boxed object:
# "cannot pass concrete object builtins.complex as builtins.object runtime input
# 0 of builtins.object.__init__". range gets away without one because its bases
# are protocols; complex's base IS object.
#
# ⛔ Seven __new__/__init__ pairs (0 args, float, int, and the four two-argument
# combinations) rather than one with a union parameter: an int argument DOES
# reach the f64 input, through int's unbox.f64 in the ABI adapter, but the
# overload is chosen by the DECLARED type first -- and a union parameter would
# arrive as a union value (tag plus lanes) instead of a number. The same asymmetry
# is why `math.sqrt(4)` already worked: a free function has one contract and the
# coercion happens at the call, while a constructor picks an overload first.
#
# ⛔ Still missing on complex: `.real` / `.imag` (attributes, not methods, so they
# need the attribute surface) and `.conjugate()`, plus complex(str).
#
# Pinned by tests/golden/cases/complex_constructor.py.
#
# ============================================================
# (46) FIXED: EIGHT MISSING math FUNCTIONS.
# ============================================================
#     import math
#     print(math.log2(8.0))
#     # module 'math' has no attribute 'log2' in this runtime
#
# ⭐ FIXED 2026-08-19. The manifest had nine functions -- the ones random.gauss
# needed -- and a probe of twenty found seventeen missing. Added the ones whose
# answer is libm's, because that is where CPython gets them too and the bits then
# agree exactly: log2, log10, exp2, atan2, fmod, copysign, degrees, radians.
#
# ⛔ Each domain/range check is CPython's, not decoration: math_1 reads errno and
# raises. log2(0.0)/log10(-1.0) name the constraint and the operand, fmod(1.0,
# 0.0) is the generic "math domain error", exp2(10000.0) is an OverflowError.
# fmod's test is CPython's own -- a NaN result from non-NaN operands -- so
# fmod(nan, 1.0) still returns nan and fmod(inf, 2.0) raises.
#
# ⛔ degrees/radians multiply by a constant rounded ONCE (what CPython does);
# dividing by pi lands an ulp away on inputs near a tie.
#
# ⭐ isclose LANDED TOO (same day, after the note below turned out to be wrong):
# the manifest CAN carry the defaults, because a manifest function may take RAW
# f64 parameters and a float argument reaches them through float's unbox.f64 --
# the road complex(...) takes. `kwonly` / `kw_names` / `kw_defaults` in the
# callable type make the tolerances keyword-only, so `isclose(a, b, 0.2)` is
# refused here exactly as CPython refuses it. The body is math_isclose verbatim.
#
# ⭐ AND THE KEYWORD PATH ITSELF, the same day: "kw names lowering is not
# keyword-aware yet" stopped EVERY keyword argument to a manifest free function,
# which made a parameter CPython declares keyword-only unreachable by
# construction. The call already carries the mapping -- the contract's arg_names
# followed by its kw_names are the parameter order, which is the order the
# manifest function declares its inputs in -- so a keyword resolves to a position
# and the positions nobody supplied stay null. The operand walk learned that a
# null source means "take this parameter's own ly.runtime.default_*", which is
# what a GAP needs and what running out at the end could not express.
#
# ⛔ Trailing nulls are popped rather than passed: the walk already fills a short
# call from the same defaults, and leaving them in would ask it to do that twice.
# ⛔ A misspelled keyword is a static refusal here (CPython raises TypeError at
# run time), and two values for one parameter is refused too.
#
# ⛔ hypot and dist stay out, now MEASURED rather than asserted: libm's hypot
# disagrees with CPython's on 32,071 of 200,000 random pairs (1 ulp, worst
# relative 2.2e-16), because CPython 3.14 accumulates in double-double instead of
# calling libm. Wiring libm would print a different last digit for one input in
# six. gcd/lcm/comb/perm/prod are int work rather than libm calls.
#
# Pinned by tests/golden/cases/math_libm_rungs.py.
#
# ============================================================
# (45) FIXED: isinstance(x, (A, B)) -- THE TUPLE FORM.
# ============================================================
#     if isinstance(v, (int, float)):
#     # second argument to isinstance must be a statically resolved class type
#
# ⭐ FIXED 2026-08-19. The target was read as ONE class, and a tuple is not one,
# so CPython's own spelling for "any of these" was refused whole. Each element
# still has to be a statically resolved class; what was missing is looking
# inside. The per-target analyses merge: any AlwaysTrue wins, AlwaysFalse drops
# out, and the UnionTest member sets union -- which is exactly what the emitted
# code already does (one py.union.test per member, ORed).
#
# ⛔ The merge is only reachable for MORE than one target: with one target the
# call goes straight to analyzeIsInstance, because the ClassTest and
# UnionClassTest kinds carry a runtime test the merge cannot combine, and
# routing the single case through the merge would change every program that has
# one. A tuple element needing such a test is refused with that reason.
#
# The narrowing follows in both directions: the true arm holds the union of the
# selected members and the false arm holds the single member none selected,
# which is what makes `x + 0.5` resolve in an else branch.
#
# Pinned by tests/golden/cases/isinstance_takes_a_tuple.py.
#
# ============================================================
# (44) FIXED: bool IS an int, TO isinstance AND issubclass.
# ============================================================
#     print(issubclass(bool, int))   # False; CPython True
#     print(isinstance(True, int))   # False; CPython True
#
# ⭐ FIXED 2026-08-19, and it is the WRONG bucket, not a refusal: both predicates
# answered through assignability and printed a wrong truth value with no
# diagnostic. Assignability is deliberately narrower -- a bool is one truth bit,
# an int is a three-value bundle, so a bool VALUE needs emitIntFromBool to be
# stored where an int is expected. The predicates ask about the CLASS hierarchy,
# where bool's base is int. One rung, added in the hierarchy direction only.
#
# ⛔ The numeric tower is not the rule and the rung is not symmetric:
# `issubclass(int, float)` stays False (CPython's answer too, though the tower
# converts), `isinstance(1, bool)` stays False, and the reverse-direction branch
# in analyzeIsInstance -- the runtime ClassTest -- still asks assignability,
# because an int value's runtime class is int.
#
# ⛔ It changes a UNION narrowing, and that is the point rather than a side
# effect: `xs: list[bool | str]` with `if isinstance(xs[0], int)` printed "no"
# before and prints 2 now. Where the union has BOTH bool and int, the narrowing
# is now `bool | int` and `+` on it is refused -- CPython would print 31 --
# because a union has no __add__. That trade is in the project's direction: the
# old answer was silently wrong for the bool member.
#
# ⛔ redcheck could not validate this golden at first: its sentinel guard read
# only the exit code, so a wrong-answer defect (exit 0, wrong stdout) looked
# like a binary that does not exhibit its defect and the run aborted. The tool
# now counts a differing .stdout as failing.
#
# Pinned by tests/golden/cases/bool_is_an_int_to_isinstance.py and the two new
# blocks in heterogeneous_container_read.py.
#
# ============================================================
# (43) FIXED: int() OVER bytes.
# ============================================================
#     print(int(b"12"))
#     # static type !py.contract<"builtins.int"> does not provide manifest
#     # method '__init__'
#
# ⭐ FIXED 2026-08-19. int(x) is intercepted before the class-instantiation paths
# claim builtins.int, and the interception knew int / bool / str / float only;
# bytes fell through to instantiation, so the diagnostic named the TARGET's
# missing `__init__` when the unsupported thing was the ARGUMENT. CPython takes
# bytes anywhere int() takes str, over the same ASCII scan.
#
# The scan was already byte-indexed (LyLong_FromStr works on memref<?xi8>), so
# nothing had to be re-derived -- but the two callers must report DIFFERENT
# reprs (b'ab' vs 'ab'), so the parse is now a shared helper returning a
# validity bit and each caller owns its raiser.
#
# ⛔ The first version of that split leaked: the helper hands back an owned zero
# on failure and the caller raised while still holding it.
# RuntimeRaisePathTest.NoOwnedObjectIsHeldAcrossARaise named both functions and
# both call edges. The release has to precede the raise -- the raise does not
# return, so a release after it is unreachable, which is exactly what the old
# single function did by building its unreachable zero AFTER the raise.
#
# ⛔ Out of scope and measured: `float(b"1.5")` is the same gap one type over
# (the float parse is unicode-indexed via __ly_unicode_get, so it needs a
# byte-indexed variant, not a shared helper), and `bytearray` is not a bound
# name at all ("unresolved name 'bytearray'"), so int(bytearray) is a missing
# type and not this defect.
#
# Pinned by tests/golden/cases/int_of_bytes_parses_digits.py.
#
# ============================================================
# (42) FIXED: `for x in G(): yield x` INSIDE A GENERATOR.
# ============================================================
#     def relay() -> Iterator[int]:
#         for x in src():
#             yield x
#     # a generator returned out of a function cannot be resumed: ... Call the
#     # generator in the for statement, bind it to a local in the same function,
#     # or return a list
#
# ⭐ FIXED 2026-08-18. The advice was ALREADY FOLLOWED -- the generator is called
# in the for statement -- which is what marked this as a defect and not a
# boundary. The same loop in a plain function and at module scope runs, and
# `yield from src()` inside a generator runs; what had no path is a nested
# generator resumed across the OUTER generator's suspensions. When the body is
# exactly one bare `yield` of the loop target, delegation is what the program
# means, and delegation has a path, so the loop is written as `yield from`.
#
# ⛔ `ast::nodeList(statement, "orelse")` returns a non-null EMPTY list for a
# for with no else, so `!nodeList(...)` never fired and the whole rewrite was
# dead. Two builds went by before instrumenting the condition showed it. Check
# emptiness, not presence.
#
# ⛔ Still refused, and both are the nested-generator frame work: a body that is
# not delegation (`yield x * 2`), and `for x in list(src(n))` INSIDE a generator
# -- which is the materialization the diagnostic suggests, and it has the same
# problem. Materializing in the CALLER works and is what the golden shows.
#
# Pinned by tests/golden/cases/for_over_a_generator_inside_one.py.
#
# ============================================================
# (41) FIXED: `yield from` OVER ANYTHING BUT A LIST LITERAL.
# ============================================================
#     def g() -> Iterator[int]:
#         yield from range(2)
#     # source generator next lowering currently supports yields whose ...
#
# ⭐ FIXED 2026-08-18 by writing it as the loop it means: `for v in X: yield v`.
# A range, a parameter's list and a str all failed while the LOOP spelling of
# each had always worked, so the gap was `py.yield.from` in the state machine and
# not the iteration. The list/tuple literal arm is untouched -- it unrolls into
# one yield per element and needs no loop.
#
# ⛔ NOT for a sub-GENERATOR, and the suite is what said so: rewriting those too
# took `generator_yield_from` and `generator_bigint` down with "a generator
# returned out of a function cannot be resumed", because the loop iterates a
# generator VALUE -- a different, separately refused shape. `py.yield.from` IS
# the delegation implementation (send and throw pass through it), so the rewrite
# is gated on the operand's type.
#
# ⭐ The gate is also the reason this is exact: over an ITERABLE, `yield from`
# evaluates to None, which is what the loop leaves behind; over a generator it
# forwards a return value, which the loop cannot.
#
# Pinned by tests/golden/cases/yield_from_any_iterable.py.
#
# ============================================================
# (40) FIXED (AS A DIAGNOSTIC): A DICT VIEW BOUND TO A NAME.
# ============================================================
#     d: dict[str, int] = {"a": 1}
#     ks = d.keys()
#     # runtime manifest has no builtins.dict.keys method
#
# ⭐ 2026-08-18. Every CONSUMING spelling works -- `len(d.keys())`,
# `sorted(d.keys())`, `list(d.keys())`, `for k in d.keys()` -- because each
# unwraps the view before emitting it. What has no representation is the view as
# a VALUE, and the old message said so in terms of the manifest, which the program
# never touched.
#
# ⛔ REFUSED RATHER THAN SNAPSHOTTED, and that is the whole decision: CPython's
# view TRACKS later mutations and `list(d.keys())` does not, so binding a list
# where the program asked for a view is a silent wrong answer the moment anything
# inserts. The diagnostic names the consuming positions, the snapshot spelling,
# and what the snapshot gives up -- which leaves the choice with the author.
#
# Pinned by tests/golden/errors/dict_view_needs_a_consumer.py.
#
# ============================================================
# (39) FIXED: `from os import path` -- AN ALIAS THAT SHADOWED A STDLIB LOCAL.
# ============================================================
#     from os import path
#     print(path.basename("a/b.py"))
#     # <stdlib>/posixpath.py:221:12: unresolved runtime binding 'path.split'
#
# ⭐ FIXED 2026-08-18, and the diagnosis is the whole story: the failure was inside
# the compiler's OWN posixpath.py, at `comps = path.split("/")` -- a str method on
# `normpath`'s parameter, which happens to be named `path`. Binding the importer's
# alias put a canonical symbol named `path` in scope while that module compiled,
# and the qualified-name route claimed the parameter's attribute chain.
#
# `import os` never collides because nothing in the stdlib is named `os`. The
# collision is what the ALIAS brings, which is why the three failing spellings all
# involve one and the working spelling does not.
#
# The rule: a local wins over an imported namespace, asked on the ROOT of the
# dotted name -- `a.b.c` where `a` is a local is a local's attribute chain whatever
# `b` is, and a qualified symbol table cannot answer it.
#
# ⛔ `import os.path` and `from os.path import basename` remain "unsupported
# import": a dotted module NAME is a separate gap in the resolver, and the note
# there records the attempts that did not close it.
#
# Pinned by tests/golden/cases/imported_namespace_versus_a_local.py.
#
# ============================================================
# (38) FIXED: A METHOD'S UNION PARAMETER CALLED WITH ONE MEMBER.
# ============================================================
#     class Box:
#         def take(self, n: int | None) -> int:
#             if n is None: return -1
#             return n
#     Box().take(None)
#     # cannot adapt runtime bundle types.NoneType with physical values ...
#
# ⭐ FIXED 2026-08-18. An inlined body binds the argument VALUE, so the parameter
# held a `literal<None>` and the body's `n is None` narrowing had nothing to
# narrow -- the union it was written against never existed at that call site. The
# FREE-function spelling works because its call emits operands against the
# declared callable, which wraps the member, and a DEFAULT of None works for the
# same reason. The inlined path wraps too now, for positionals, keywords and
# keyword-only parameters.
#
# ⭐ `collections.Counter.most_common(None)` is this defect reached through the
# SHIPPED STDLIB -- the parameter is `int | None` and passing None explicitly is
# how CPython's own signature is exercised. A defect that only a library call
# reaches is the argument for gridding stdlib usage, which this file already
# records under the sweep notes.
#
# Pinned by tests/golden/cases/union_parameter_takes_a_member.py.
#
# ============================================================
# (37) FIXED: A float CLASS ATTRIBUTE WITH AN int INITIALIZER.
# ============================================================
#     class P:
#         v: float = 1
#     print(P.v)
#     # RuntimeError: module global 'P.v' referenced before assignment
#
# ⭐ FIXED 2026-08-18 by giving this channel the refusal the MODULE-GLOBAL write
# already makes. `x: float = 1` at module scope says so at emit -- `coerceValue`
# declines to retype between the numeric contracts, and the write reports it --
# while the class-attribute cell, the same storage under the same rule, had no
# check at all: the store of an int into a float cell was dropped further down,
# leaving the cell unassigned and the failure to the reader at RUNTIME, naming an
# internal cell name.
#
# ⛔ Still a deviation from CPython, and the same one the module global documents:
# CPython prints 1 there, because its annotation is inert and the value stays an
# int. A cell whose representation is fixed by the declaration cannot hold that,
# so the answer is a refusal that names the attribute and what to write.
#
# Pinned by tests/golden/errors/class_attribute_numeric_representation.py.
# ⛔ redcheck cannot red-check an errors golden; `run_case.py --expect-layer emit`
# fails on the pre-fix binary, which reaches the RuntimeError instead.
#
# ============================================================
# (36) FIXED: `*args` ON A METHOD, AND AN EMPTY EVIDENCE SEQUENCE.
# ============================================================
#     class Registry:
#         def many(self, *items: str) -> int:
#             return len(items)
#     Registry().many("p", "q")
#     # too many positional arguments for inlined class method
#
# ⭐ FIXED 2026-08-18. The free-function spelling always worked: a real function
# binds its vararg to the tuple the call packed, and the inlined method path had
# no such step -- it walked the declared positionals and refused the rest.
#
# ⭐ THE EMPTY CASE IS A SECOND DEFECT, and it only showed once the first was
# fixed: `R().tag("p")` binds an empty tuple and ITERATING it reported "list
# iteration evidence match/value count mismatch". An evidence sequence with no
# elements has nothing to select between, and `valid` is already false there, so
# it now iterates zero times and binds a dead placeholder for the element the op
# must still produce. `for x in ()` at module scope always worked because a
# literal empty tuple takes the RUNTIME path, where a length of zero is ordinary.
#
# ⭐ `**kwargs` CAME WITH IT, in the same round: the unmatched keywords are
# collected into the dict the callee would have received, built through
# `LyValueRef` because the values are already EMITTED and a dict literal is
# written in AST -- the machinery the augmented-assignment rewrite uses to name a
# subexpression it must not evaluate twice. The guess recorded here about which
# mechanism it would need turned out to be the right one.
#
# ⛔ One thing it did NOT fix:
#   `self.xs = list(xs)` with no field annotation -- the field reads as
#     `builtins.object` OUTSIDE the class, so `len(r.xs)` is refused there while
#     the body's own `len(xs)` is fine. The class-field pre-pass types fields
#     without a call site and a vararg has no type until one exists. Annotating
#     the field is the working spelling.
#
# Pinned by tests/golden/cases/method_takes_star_args.py.
#
# ============================================================
# (35) FIXED: extend/join TAKE ANY ITERABLE, NOT ONLY A LIST.
# ============================================================
#     xs: list[int] = []
#     xs.extend((1, 2))
#     # cannot adapt builtins.tuple to runtime input 1 of builtins.list.extend
#     #   [values: 'memref<14xi64>', expected 'memref<9xi64>']
#
# ⭐ FIXED 2026-08-18, as the rewrite a GENERATOR argument already took: the callee
# consumes the whole iterable, so `list(...)` around it is exact.
#
# ⭐ THE TRIGGER TOOK THREE READINGS, and the first two were wrong in an
# instructive way:
#   1. "materialize when the call does not resolve" -- it resolves. The manifest
#      declares the parameter as the PROTOCOL `Iterable`, so a tuple type-checks.
#   2. "materialize when the declared parameter is builtins.list" -- it is not; the
#      declared parameter IS the protocol. Instrumenting the condition is what
#      showed it: `declared=!py.protocol<"Iterable", [int]> actual=tuple[int]`.
#      The refusal comes from the runtime ABI, which implements the list case only.
#   3. the protocol itself is the trigger.
#
# ⛔ EXCEPT an argument of the RECEIVER's own contract: `s.update(other_set)` and
# `xs.extend(other_list)` are the shapes the runtime implements directly, and
# materializing those would break a working call.
#
# ⛔ `s.update((1, 2))` stays refused: the rewrite hands set.update a list and its
# runtime wants a set. It was refused before, so nothing regressed -- closing it
# needs the manifest to implement the other cases.
#
# Pinned by tests/golden/cases/extend_takes_any_iterable.py.
#
# ============================================================
# (34) FIXED: A METHOD'S DEFAULT IS EVALUATED ONCE.
# ============================================================
#     class Bag:
#         def add(self, into: list[int] = []) -> int:
#             into.append(1)
#             return len(into)
#     print(Bag().add(), Bag().add())   # printed 1 1; CPython prints 1 2
#
# ⭐ FIXED 2026-08-18. The FREE-function spelling was already right, which is the
# whole localisation: the evaluate-once cell was gated on the def being a direct
# child of the MODULE body, so a method fell to the per-call provider -- a fresh
# list every call, and a side-effecting default (`n: int = make()`) firing again
# each time.
#
# Two halves, and the first alone was not enough (it added a THIRD "eval" rather
# than removing the second):
#   1. register the cell under the CLASS statement, because the module walk
#      flushes pending cells at the statement it skipped. The note at that call
#      site already promised this ("method defaults registered under a class
#      statement flow through the same cells") and nothing had ever registered one.
#   2. make the INLINED call read the cell instead of re-emitting the expression.
#      An inlined method has no call for the callable's default-value attributes
#      to serve, so it reads the cell by name.
#
# ⛔ NOT `markBoxedModuleGlobal` on that read: a default cell is not a module
# global even though both are py.global.get/set, and the lowering says so ("this
# population is never marked `ly.global.boxed`, so an int default stays in the
# native word cell"). Marking it produced "module global ... referenced before
# assignment" for an int default -- the store had gone to the other cell.
#
# ⛔ A @classmethod's default is still per-call: its body is emitted from a node
# that is not the one in the class body, so the scan never registers a cell for
# it. @staticmethod and keyword-only defaults are fixed and are golden sections.
#
# ⛔ AND A COLLISION FOUND WHILE WRITING THE GOLDEN: a module-level
# `def free(...)` stops the whole program with "redefinition of reserved function
# 'free' of different type is prohibited", 34 times over, because the name meets
# the C library symbol. Any Python program may define a function called `free`;
# the emitted symbol should not be its bare name. Not fixed -- recorded here.
#
# Pinned by tests/golden/cases/method_default_evaluated_once.py.
#
# ============================================================
# (33) THE STANDING WORK LIST: wb_sweep_findings_2026_08_18.py
# ============================================================
# Forty agents over ten domains, 30 verified findings, 27 still live after this
# session's twelve rounds. They are written out one per entry -- program, both
# outputs, a working neighbour, the verifier's notes -- in
# tests/probe/wb_sweep_findings_2026_08_18.py, ordered silent-wrong-answer,
# crash, false-refusal, missing-feature.
#
# ⭐ Re-checked against the current binary rather than trusted: 3 of the 30 were
# already closed by this session and are not repeated. Do that again before
# picking -- and note that finding 8's SYMPTOM had moved (a mis-raised IndexError
# became an ownership error), which only bisecting the session's saved pre-fix
# binaries showed was not a regression.
#
# ⭐ The four the next round should start from, by value:
#   a loop-carried accumulator resets across a `yield` in a for-loop over a LIST
#     (silent; `range` works, `["a"]` works, so it is same-contract lives at the
#     suspension -- and with two elements it becomes a garbage bigint)
#   `for i in range(3): ...` then `print(i)` -- unresolved name (the loop target
#     is not readable after the loop; CPython leaves it bound, and the empty-loop
#     case is a NameError no static binding can express, which is the open part)
#   an `except` handler ending in `return` is not a terminator, so a name bound
#     only in the try body reads as unresolved
#   two SIGSEGVs in the ownership/classes domains, each with a one-edit neighbour
#
# ============================================================
# (32) PARTLY FIXED: A CONTAINER OF INTS WHERE FLOATS ARE DECLARED.
# ============================================================
#     def f() -> list[float]:
#         return [1]
#     print(sum(f()))              # printed 5e-324; CPython prints 1
#     class C:
#         def __init__(self) -> None:
#             self.xs: list[float] = [1]
#     print(C().xs[0])             # printed 5e-324
#     t: tuple[float, float] = (1, 2)
#     print(t[0] + 0.5)            # printed 0.5; CPython prints 1.5
#     xs: list[float] = [1]        # module scope
#     print(xs[0])                 # printed 5e-324
#
# ⭐ FIXED 2026-08-18 for the RETURN, the FIELD and the dict-in-a-function paths:
# `coerceValue` no longer retypes a container between two numeric element
# contracts. It already declined the SCALAR retyping for the same reason ("int,
# float and bool share no representation") and the container case is that lie one
# level in -- the element type IS the storage, so int boxes sat in float slots.
#
# ⭐ THE MEASUREMENT THAT SETTLED THE POLICY. Two of these shapes printed
# CPython's answer, so refusing looked like giving back working ground -- until
# the same program was asked to DECODE an element:
#
#     print(t)          -> (1, 2)     matches CPython
#     print(t[0] + 0.5) -> 0.5        CPython 1.5      SAME PROGRAM
#     return [1]        -> [1]        matches CPython
#     sum(f())          -> 5e-324     CPython 1        SAME PROGRAM
#
# Nothing was working; nothing had decoded yet. A refusal is strictly better than
# either half.
#
# ⛔ Why NOT convert to 1.0: CPython does not convert at an annotation, so it
# would print 1.0 where CPython prints 1 -- and the first attempt at this fix DID
# convert at the container-literal element site, which regressed `[1.0, 2]` from
# CPython's `[1.0, 2]` to `[1.0, 2.0]` and the tuple from `(1, 2)` to
# `(1.0, 2.0)`. Reverted. The `float | int` union element that a MIXED literal
# already builds is the representation that would print CPython's answer for all
# of them, and giving an ALL-int literal that representation under a float
# expectation is the open question.
#
# ⛔ TWO CHANNELS STILL MIS-EXECUTE, each measured:
#   `xs: list[float] = [1]` at MODULE scope     -> 5e-324. Goes through the static
#       attribute initializer (`ly.module_static_attr_values`), not coerceValue.
#       The class-attribute spelling of it (`class P: v: float = 1`) is the same
#       channel and RuntimeErrors with "referenced before assignment" instead.
#   `t: tuple[float, float] = (1, 2)`           -> t[0] + 0.5 gives 0.5. The tuple
#       literal builds `tuple[float, float]` directly from the positional
#       expectations, so there is no retyping for coerceValue to decline.
#
# Pinned by tests/golden/errors/int_element_in_a_float_container.py. ⛔ redcheck
# cannot red-check an errors golden (it asks whether the CASE fails), so the check
# was run with the real runner: `run_case.py --expect-layer lower` FAILS on the
# pre-fix binary, which exits 0 and prints 5e-324.
#
# ============================================================
# (31) FIXED: STORING INTO A LIST HELD IN A FIELD.
# ============================================================
#     class Box:
#         def __init__(self) -> None:
#             self.items: list[int] = []
#     b = Box()
#     b.items.append(1)
#     b.items[0] = 9
#     # IndexError: list assignment index out of range
#
# ⭐ FIXED 2026-08-18. `sequenceEvidenceBacked` is a flag about the container's
# KIND, not a promise that the bundle describes the contents, and a field read
# strips the contents it did know. Zero recorded elements read as length zero, so
# an in-range store raised; `del b.items[0]` had it too.
#
# ⭐ AND THE SEEDED CASE, which the empty one only hinted at: a field seeded
# `[0]` and grown by one append still describes one element, so `items[1] = 9`
# took the evidence arm with stale contents and double-booked the slot ("owned
# resource ... released or transferred more than once"). The rule is one rule --
# the evidence tier is sound only where the walk sees EVERY mutation, and through
# a field it cannot, because each read builds a fresh bundle from the owner. So an
# interior view stores through the payload and a local keeps the evidence arm.
#
# ⭐ THE LEAK THE FIX EXPOSED, and it was NOT caused by it: routing field stores
# onto the runtime path made `xs[5] = 9` inside a try leak 52 B -- and the same
# program on a LOCAL list leaked 52 B on every binary in this session. The caller
# retains the value for the slot and hands `LyList_SetItemBox` a box that owns
# that reference; the raise happened inside the shared index normalizer, which has
# no box to release. SetItemBox now range-checks itself, releases the box, then
# raises. The delete path keeps the shared normalizer: it has no value box.
#
# ⛔ "Exposed" and "caused" are different, and the leak gate is what told them
# apart: measured on the pre-fix binary for the LOCAL spelling, which the fix does
# not touch.
#
# Pinned by tests/golden/cases/store_into_a_field_list.py.
#
# ============================================================
# (29) FIXED: A SLICE OF INSTANCES WAS TYPED AS ITS ELEMENT.
# ============================================================
#     class V:
#         def __repr__(self) -> str: return "v"
#     vs = [V(), V()]
#     print(vs[0:2])      # printed v; CPython prints [v, v]
#
# ⭐ FIXED 2026-08-18. Byte-identical to `print(vs[0])`: the print/repr dispatch
# asked the LENIENT inference walk, got `V` -- the element -- found `V.__repr__`
# and inlined it with the slice's list handle as the receiver. The element's
# `__repr__` ran exactly once for a three-element slice, `vs[0:0]` printed an
# element where CPython prints `[]`, `list(vs[0:2])` did not compile, and
# `vs[0:2].k` reached the lowering as a field read.
#
# ⭐ THE STRICT WALK WAS ALREADY RIGHT, and the note there said why the lenient
# one was left wrong: correcting both made `a[bump():3] += [99]` print
# `[1, 2, 3, 4, 5]` instead of `[1, 2, 3, 99, 4, 5]`. That deferred question has an
# answer now, and it is not about the type: a slice target's `+=` is a slice
# ASSIGNMENT in CPython, reading `a[i:j]` produces a NEW list, so the in-place
# rewrite `a[i:j].extend([99])` extends a copy. The route does not want a
# different type, it wants not to run -- so it declines slice targets and the
# lenient walk is corrected for everyone.
#
# ⛔ `list[int]` slices always printed correctly, which is why this survived: an
# int element reaches no source-class method, so the wrong answer selects nothing.
# It takes a user class to become visible. The same reason the first fix attempt
# (re-inferring strictly at the print/repr site only) was thrown away: it left
# `list(vs[0:2])` and the attribute path still wrong, and the root was one answer
# in one walk.
#
# ⛔ Found by a 40-agent parallel sweep over ten domains, then reproduced by hand
# before touching anything. 43 of 48 programs in the earlier serial sweeps had
# agreed; this needed a domain (classes) crossed with a construct (slicing) that
# neither sweep had crossed before.
#
# Pinned by tests/golden/cases/slice_of_instances_is_a_list.py, whose
# augmented-slice sections are the regression guard for the deferred question, and
# by the existing augmented_assignment_evaluates_once.
#
# ============================================================
# (30) OPEN, MEASURED: PRINTING AN INSTANCE OF A CLASS WITH NO __repr__
#      INSIDE A CONTAINER ABORTS.
# ============================================================
#     class V:
#         def __init__(self, n: int) -> None:
#             self.n = n
#     print([V(1)])
#     # cf.assert: repr: boxed element has no conforming __repr__
#     # exit code 134, with an LLVM stack trace
#
# CPython prints `[<__main__.V object at 0x...>]`. And Lython already prints
# `<__main__.V object at 0x...>` for the SCALAR -- `print(V(1))`, `repr(V(1))` and
# `str(V(1))` all match CPython's form, class name included, through
# `materializeDefaultObjectRepr`. So the machinery exists and the container's
# element dispatch cannot reach it: `generateBoxedMethodHook` registers a class id
# only when `classMethodSymbol(classOp, "__repr__")` finds a symbol, and a class
# with no `__repr__` anywhere in its MRO has none, so the table has a hole and the
# element walk asserts.
#
# ⛔ A class that defines only `__str__` aborts the same way (the hook is keyed on
# `__repr__`, and CPython also uses `__repr__` for elements inside a container, so
# the abort -- not a `__str__` fallback -- is the shape to fix).
#
# ⛔ Three ways to close it, and the choice is not obvious:
#   1. synthesize a per-class default-repr function and register it in the hook --
#      matches CPython including the class name, and is the most work;
#   2. fall back to `LyObject_BoxedRepr`'s existing default arm -- one edit in
#      builtins.mlir, but its prefix is the generic `<object object at 0x`, so the
#      class NAME would be wrong, which is a silent wrong answer;
#   3. refuse at emit when the element contract is statically a source class with
#      no `__repr__` -- the earliest static boundary, and it rejects a program
#      CPython runs.
# Not shipped: an abort is worse than all three, but picking between them is the
# feature-boundary question this file already records.
#
# ============================================================
# (28) FIXED: `C.__name__`. AND WHAT IT MEASURED ABOUT THE TYPE-OBJECT SURFACE.
# ============================================================
#     class C: pass
#     print(C.__name__)
#     # attr.get type object has no static runtime attribute '__name__'
#
# ⭐ FIXED 2026-08-17 as a fold to a string constant -- the last dotted component
# of the contract name, so `int` answers "int" and not "builtins.int".
#
# ⭐ TWO CHANNELS, and the first attempt only did one. The emitter's fold was
# invisible to every consumer that asks the TYPE first: `[C.__name__,
# Base.__name__]` joined to `list[object]` ("a type-erased `object` value cannot
# be stored in a runtime container slot"), because a literal's join is computed
# from the inferred element types and not from the emitted values. Same shape as
# the three-channel note on `TypeSystem::inferExpr` elsewhere in this file: an
# emitter fold with no inference arm is half a feature.
#
# ⛔ THE REST OF THE TYPE-OBJECT SURFACE IS STILL REFUSED, measured 2026-08-17:
#
#   print(int)               -> runtime method receiver has no concrete contract
#   int is int               -> `is` requires reference-typed operands that
#                               resolve statically
#   c.__class__              -> class C has no field '__class__'
#   type(x)                  -> unresolved name 'type'
#   list.__name__            -> unresolved name 'list'  (the CONTAINER builtins
#                               are not bound as names at all; int/str/float/bool
#                               are)
#
# So `type(x) is C`, which is what the sweep found, needs a type-object VALUE and
# not another fold. Recorded rather than half-built.
#
# Pinned by tests/golden/cases/class_name_attribute.py.
#
# ============================================================
# (27) FIXED: break/continue INSIDE A TRY, IN A LOOP THAT CARRIES A LOCAL.
# ============================================================
#     total = 0
#     for s in ["1", "x", "3"]:
#         try:
#             total += int(s)
#         except ValueError:
#             continue
#     # break/continue through try/finally in a loop with carried (reassigned)
#     # locals is not implemented yet
#
# ⭐ FIXED 2026-08-17. That is the canonical parse-and-skip loop; the refusal
# covered every accumulator loop with a jump inside a try. The SAME statement
# without the accumulator always compiled, which is what named the piece: the
# completion branches emitted after `py.try` forward the loop's carried operands,
# and in SSA they could only forward pre-try values.
#
# The fix is that they stop reading SSA. A local the try body rebinds is already
# promoted to an R6 cell for the extent of the statement -- and the promotion was
# extended to loop-carried names earlier, with its own note -- so the branches
# LOAD the cell. Two pieces:
#
#   1. the completion branches load and forward (EmitterExceptions.cpp)
#   2. the operand count is asked of the TARGET, not of the carried set: both
#      completion checks are emitted whenever either jump appears, and a loop with
#      no `break` gives its after-block no arguments. Without that, every
#      `continue`-only program failed with "branch has 1 operands for successor
#      #0, but target block has 0".
#
# Measured: 12 shapes (for/while x break/continue x body/handler/finally), 5 with
# an OWNED accumulator (str and list), all agreeing with python3.14 and all
# net 0 allocs / 0 B on the leak gate.
#
# ⛔ `continue` inside a `finally` behaves correctly but CPython 3.14 prints
# "SyntaxWarning: 'continue' in a 'finally' block" and Lython prints nothing, so
# that spelling is out of the golden. Lython emits no SyntaxWarnings at all --
# one item, not one per warning.
#
# ⛔ TWO NESTED SHAPES STAY REFUSED, and finding them is the reason to grid a fix
# outward as well as inward. Both were refused before this fix; what changed is
# that the message names the shape:
#
#   nested loops, `total` carried by BOTH, `continue` in the inner try
#       -> "ownership CFG exploration exceeded 20000 states" on the first cut of
#          this fix. A resource limit is not a diagnostic, and the shape was a
#          clean refusal before, so it is refused again -- by a guard that asks
#          for a CONTINUE and for two or more carriers, because the three
#          neighbours all compile: the same program with `break`, an accumulator
#          carried by the inner loop alone, and a `continue` outside the try.
#
#   nested try whose OUTER arm has a finally
#       -> the inner try promotes the name, the outer statement sees it already
#          rebound and does not, and the outer completion branch has nothing to
#          forward. This is what reaches the residue guard below.
#
# ⛔ The residue guard was written for in-place mutation receivers, and those do
# NOT reach it: four attempts (`xs.append(v)`, `xs += [v]`, each with and without
# `xs = xs + [v]` to force the carry) all compiled, because an in-place mutation
# is not a reassignment and the name never enters the carried set. The nested-try
# shape above is what reaches it.
#
# Pinned by tests/golden/cases/loop_control_inside_try.py.
#
# ============================================================
# (25) FIXED: `assert`.
# ============================================================
#     assert n > 0, "must be positive"
#     # emit error: unsupported statement kind 'Assert'
#
# ⭐ FIXED 2026-08-17, as a rewrite: `if not test: raise AssertionError(msg)`.
# Nothing new reaches the dialect, and both halves already worked --
# `builtins.AssertionError` was in the runtime taxonomy and `raise E` with no
# arguments had already been fixed on the Raise arm.
#
# ⛔ Not elided under any flag: CPython drops asserts under -O, Lython has no -O,
# and the CPython DEFAULT is that they run.
#
# ⛔ An uncaught assert's traceback still differs from CPython's, and not because
# of assert: CPython underlines the failing expression with `^^^^` markers and
# Lython prints no caret line for any statement. Same difference shows on
# `p.give(1)` and every other frame, so it is one item, not one per statement.
# The golden catches its failures for that reason.
#
# Pinned by tests/golden/cases/assert_statement.py.
#
# ============================================================
# (26) OPEN: THE FOUR REFUSALS THE 2026-08-17 SWEEP FOUND BESIDE assert.
# ============================================================
# Six batches of eight realistic programs; 43 of 48 agreed with python3.14 on
# stdout AND exit code. What the other five were:
#
#   `fs = [outer(i) for i in range(3)]; [g(10) for g in fs]`
#       -> TypeError: callable target is not available (at RUNTIME, not a
#          diagnostic). `add5 = outer(5); add5(1)` works, so it is a closure
#          stored in a CONTAINER and called through the element read. A refusal
#          that only appears at runtime is the worst kind here -- it should be an
#          emit diagnostic at least.
#
#   `[*vals, 4]` and `{**d, "b": 2}`
#       -> unsupported expression kind 'Starred'. Feature.
#
#   `type(1) is int`
#       -> unresolved name 'type'. Feature.
#
#   `isinstance(other, Vec)` on an `object` parameter, inside `__eq__`
#       -> "isinstance on an object-typed value requires dynamic object
#          inspection". Already recorded; this is the shape that makes it matter,
#          because `__eq__`'s signature is `object` by convention.
#
#   a subclass overriding a method, called through the base type
#       -> "'area' is overridden by a subclass of 'Shape', so this call cannot be
#          resolved from the static type of the receiver". This is the DESIGN
#          boundary (no dynamic dispatch), not a defect.
#
# ============================================================
# (23) FIXED: AN UNBOUND BASE METHOD DID NOT ACCEPT `self`.
# ============================================================
#     class Child(Base):
#         def __init__(self, n: int) -> None:
#             Base.__init__(self, "c")
#     # argument 'self' of '__init__' is declared Base and this call gives it
#     # Child
#
# ⭐ FIXED 2026-08-17. `Base.__init__(c, "z")` at module scope worked and
# `Base.greet(c, "x")` worked, which is what named the cause: by then
# `py.class @Child` is in the module with its `mro_names`, and inside Child's own
# method it is not, so the assignability walk found no bases and called a subtype
# unrelated. The emitter's `classMros` has the hierarchy before any body is
# emitted; the check reads it when the module cannot answer.
#
# ⛔ Still refused, correctly: a base method that assigns a field the BASE does
# not declare is "class Base has no field 'name'". Fixing the MRO check moved
# that program from a wrong diagnostic to the right one.
#
# Pinned by tests/golden/cases/unbound_base_method_takes_self.py.
#
# ============================================================
# (24) OPEN, LOCALISED: A NARROWED UNION FIELD RETURNED FROM A METHOD.
# ============================================================
#     class Opt:
#         def __init__(self) -> None:
#             self.value: int | None = None
#         def get_or(self, default: int) -> int:
#             v = self.value
#             if v is None:
#                 return default
#             return v            # <-- this
#     print(Opt().get_or(7))
#     # error: owned resource from builtin.unrealized_conversion_cast result 0
#     # reaches function exit without release, transfer, or owned return
#
# The measurements that place it (2026-08-17):
#
#   the same code as a FREE function taking the object      -> works
#   passing `c.v` to a free function that narrows           -> works
#   `return v + 0` / `return str(v)` / `return 0`           -> works
#   returning a NON-union field (`return self.name`)        -> works
#   `if v is not None: return v` (arms swapped)             -> same failure
#   `str | None` instead of `int | None`                    -> same failure
#
# So it is not narrowing, not the union, and not the return: it is returning the
# union field's LANES. A union field is the one shape that still takes the
# pre-4a lane splice in `lowerAttrSet` ("Union fields are the only shape that
# reaches it"), so the instance's owned-local marker COVERS the member's lanes --
# the runtime-lowering dump shows the marker as
# `(memref<16xi64>, i64, memref<2xi64>, memref<2xi64>, memref<?xi32>)` -- and the
# non-None arm forwards those very values out as the result. The pass then sees
# the marker's own values escape and reports the marker unreleased.
#
# ⛔ Two fixes, both bigger than a patch: box-front the union field like every
# other object field (the kernel change the residual path is waiting for), or
# retain the member where a narrowed union escapes a frame. This is the same
# family as the recorded `float | int` return, and the same conclusion.
#
# ============================================================
# (22) FIXED: THE LOWERING READ ITS OWN FREED BUNDLE.
# ============================================================
#     class Pool:
#         def __init__(self) -> None:
#             self.free: list[int] = [1, 2, 3]
#             self.used: list[int] = []
#         def take(self) -> int:
#             v = self.free.pop()        # <-- reported here
#             self.used.append(v)
#             return v
#         def give(self, v: int) -> None:
#             self.used.remove(v)
#             self.free.append(v)
#     p = Pool(); p.take(); p.give(3); print(p.free, p.used)
#     # cannot adapt builtins.list to runtime input 0 of builtins.list.__len__
#     # [values:, expected 'memref<9xi64>']
#
# ⭐ FIXED 2026-08-17, and it is the defect in this file whose LOCATION lies the
# most. `lowerBoundMethodCall` took its receiver as `const RuntimeBundle &`, the
# caller passed a reference INTO `valueBundles`, and the function inserts into
# that DenseMap on nearly every path. A rehash moves the entry; the liveness pin
# emitted after the pop then read freed memory.
#
# ⛔ NEIGHBOURS THAT COMPILED, all of them one edit from the failing program.
# Each is a bundle-count change, not a semantic one:
#
#   drop `self.used.remove(v)` from give                -> compiles
#   drop `self.free.append(v)` from give                -> compiles
#   swap those two statements                           -> compiles
#   declare `used` before `free`                        -> compiles
#   `self.used: list[int] = [0]` instead of `[]`        -> compiles
#   `print(p.free)` or `print(p.used)`, not both        -> compiles
#   never call `p.give`                                 -> compiles
#   two more classes in the same FILE                   -> compiles
#
# The last one is why the golden is small and its shape exact: the first version
# of tests/golden/cases/two_fields_mutated_in_two_methods.py had three classes
# and redcheck reported "GREEN <-- cannot be made to fail".
#
# ⭐ The general lesson, and the reason this is worth the space: NO AMOUNT OF
# READING THE REPORTED LINE CAN LOCALISE THIS. What found it was gridding one
# axis at a time and noticing that every neighbour compiled -- a defect whose
# trigger is "how many bundles exist" has no cause at the failing statement.
#
# An audit for the same mechanism (a `const RuntimeBundle &` parameter that
# aliases the map, used after an insertion) found six more functions:
# lowerDictEvidenceGetItem, materializePayloadObjectBundle,
# materializeObjectBundleForStorage, lowerFunctionTargetCall,
# lowerSourceGeneratorNext, lowerStaticCtypesCall. None of them rewrites its own
# operand key, so a local copy is a pure UB fix; all six now copy. ⛔ The audit
# is not a proof of absence: it only covers parameters spelled
# `const RuntimeBundle &`, and only where an insertion is syntactically visible
# in the same function.
#
# ============================================================
# (21) FIXED: A LOCAL STORED INTO A FIELD WAS FREED AT THE STORE.
# ============================================================
#     seed = [3, 1, 2]
#     b = Bag(seed)            # __init__ does self.xs = xs
#     for v in seed:           # SIGSEGV
#     [v for v in seed]        # []
#     max(seed)                # ValueError: max() iterable argument is empty
#     while i < 3: print(len(seed))   # 0 0 0
#
# ⭐ FIXED 2026-08-17. A use-after-free, and the most valuable thing about it is
# how it hid. The field store retains for the slot and releases the value's own
# token -- correct for a temporary, since the two cancel and the slot inherits
# the single reference. For a local the caller keeps reading, the entity is freed
# at the store, and the read finds whatever the NEXT allocation wrote there. So:
#
#   `print(len(seed))` at module scope right after the store  -> 3   (nothing
#                                                                    allocated)
#   `print(len(seed), len(b.xs))`                             -> 3 3
#   a later `print(b.xs)` anywhere after the loop             -> whole program
#                                                                right
#   the same read with an allocation in between               -> 0
#
# Three of those four make the defect invisible, and two of them are what you
# would naturally write to probe it. ⭐ THE PROBE THAT FOUND IT was the exit
# code: rc=139 on `for v in seed`, which no amount of stdout-diffing sees.
#
# ⛔ The test is "does this store dominate a use of the value", asked on the
# PY-LEVEL operand. Asking the bundle's physical values answers "nothing
# outlives this" for every shape above: the walk lowers in program order, so a
# later read is still an unlowered `py.len` and the handle has no uses yet.
# Measured -- the first version of the fix used physicalValues() and changed
# nothing at all.
#
# The read side of the same alias (12 shapes) needed `sharedWithHolder` on three
# more read paths: `d[k]` for a literal key (KeyError for a key added through the
# other name) and `x in xs` (constant-folded against the pre-store slots), on top
# of the sequence `[i]` from (20).
#
# Pinned by tests/golden/cases/field_store_keeps_the_local_alive.py and the
# extended mutation_through_an_alias.py.
#
# ============================================================
# (20) FIXED: STRUCTURAL MUTATION THROUGH A NON-LOCAL RECEIVER,
#      AND THE ALIAS READ UNDER IT.
# ============================================================
#     self.seen.add(n)         # set.add requires a rebindable local receiver
#     self.rows.insert(i, v)   # list.insert requires a rebindable local ...
#     self.rows[:n] = [9]      # slice assignment requires a named local list
#     del self.rows[:n]        #   target (field containers are not supported)
#
# ⭐ FIXED 2026-08-17. Four refusals, one expired premise: "a mutation may
# reallocate, so it hands back a re-description only a local can hold".
# `LySet_AddBox`, `LyList_EnsureCapacity`, `LyList_SetSlice` and
# `LyList_DelSlice` are all VOID now -- the growth writes the new items address
# through the handle. Checking the manifest signature rather than reasoning
# about the era is what found all four at once.
#
# Two further defects fell out of the grid rather than out of the refusals, and
# both were SILENT:
#
#   `xs = [1, 3]; xs.insert(1, 2); print(xs[1])` printed 3 (CPython: 2), and
#   three live reads aborted the ownership verifier. An insert shifts every slot
#   at or past the index; `__setslice__` already demoted its evidence for
#   exactly this and `insert` did not.
#
#   `b = Bag(seed); b.xs[0] = 9; print(seed[0])` printed 3 (CPython: 9) -- the
#   alias read. `b.xs.sort()`, `holder[0][0] = 9` and `by_name["a"][1] = 9` were
#   wrong the same way, and `b.xs.append(9); seed[3]` did not compile. The mark
#   (`sharedWithHolder`) was already set at every absorption and nothing
#   consulted it on the read side.
#
# ⛔ The alias fix is the READ moving to the payload, not the evidence being
# dropped: dropping it at the absorption was measured three ways and takes
# 145-146 tests down, because the evidence is where the slot's owned reference
# is booked.
#
# The 75-cell grid (15 mutators x 5 receiver kinds: local, field, field from
# outside, class attribute, container element) is clean.
#
# Pinned by tests/golden/cases/structural_mutation_through_a_view.py,
# list_insert_shifts_the_reads.py, mutation_through_an_alias.py (which replaces
# errors/list_insert_on_field, whose own note asked to be retired this way).
#
# ============================================================
# (19) FIXED: A CLASS ATTRIBUTE HOLDING A CONTAINER.
# ============================================================
#     class R:
#         items: list[str] = []
#     print(R.items)
#     # unsupported static class attribute expression for 'items'
#
# ⭐ FIXED 2026-08-17, and it took four attempts that each hit a different wall
# -- worth keeping, because the first three readings all looked right.
#
#   1. "give it a cell like a module global" -- the cell mechanism already
#      existed (`classAttrSlots`), so this was never the work.
#   2. "the exclusion's reason is stale, so widen the predicate" -- true (the
#      note said container cells go stale against reallocation, and
#      collectModuleGlobals stopped excluding containers when the measurement
#      showed the growth writes THROUGH the handle) but not sufficient.
#   3. Widening it broke the runtime lib: `ctypes.Structure._fields_` is a list
#      the COMPILER consumes, and slotting it emits a module-level store, which
#      a runtime-internal lib module may not have. `_dunder_` names stay on the
#      constant channel.
#   4. "and the read path does not consult the slots" -- WRONG, and this is the
#      reading to be most careful of. It does; the earlier probes failed for
#      reason 3, and once that was excluded both the class-object read
#      (`R.items`) and the instance read (`r.items`) answered.
#
# ⛔ What is still refused: `cls.tags.add(x)` on a SET attribute -- "set.add
# requires a rebindable local receiver", because that mutation rebinds and the
# receiver is an attribute read rather than a name. The same set as a MODULE
# global works, where the receiver is a name. `list.append` and `d[k] = v` are
# unaffected: they write through the handle. Pinned by
# tests/golden/cases/class_attribute_container.py.
#
# ============================================================
#     class R:
#         items: list[str] = []
#     print(R.items)
#     # unsupported static class attribute expression for 'items'
#
# ⭐ MEASURED BOUNDS, 2026-08-17. The four SCALAR kinds work
# (`n: int = 0`, `s: str = "a"`, `f: float = 1.5`, `b: bool = True`) and
# `cls.n += 1` in a classmethod works. Every container refuses -- list, dict and
# even a `tuple[int, int]` -- on the READ, before any mutation. `R.items = [...]`
# has its own message ("class static attribute mutation is not supported").
#
# The channel is the reason: `classStaticValue` stores the attribute as a
# compile-time constant EXPRESSION and `lowerAttrGet` re-materializes it per
# read, with arms for constant.none/bool/int/float/str and nothing else. A
# container cannot be re-materialized per read, because every read has to be the
# SAME object -- a mutation through one read must be visible through the next.
#
# The precedent is the module-global container, which was closed the same way it
# would have to be here: give it a CELL and let the reads hand back the handle
# (`16c2b736 feat(globals): a container module global has a cell, because the
# handle is what a cell holds`).
#
# ⭐ THE MECHANISM ALREADY EXISTS AND IS ONE PREDICATE AWAY -- and the predicate
# is not the whole change. `EmitterClasses.cpp` has `classAttrSlots`: "attributes
# of main-module classes whose widened type has module-global cell storage become
# slot-backed (reads and writes go through the cells; the initializer expression
# is no longer restricted to constants). Container-typed attributes stay on the
# constant channel: their storage cells would go stale against reallocation, the
# same reason collectModuleGlobals excludes them."
#
# ⛔ THAT REASON IS STALE -- `collectModuleGlobals` does NOT exclude containers
# any more ("Every contract gets a cell; the file header has the measurement that
# replaced the container exclusion"), because the growth writes THROUGH the
# handle. But widening the `storable` predicate to list/dict/set/tuple was tried
# on 2026-08-17 and does not land:
#
#   - the runtime lib stops building: `stackguard_support.py:96` becomes
#     "runtime lib module must not run module-level code; only imports and
#     function definitions are allowed", because a slot-backed attribute needs an
#     initialization statement and a runtime-internal module may not have one;
#   - and the user programs still refuse, so the READ path does not consult the
#     slots for these -- `lowerAttrGet`'s constant channel is still what answers.
#
# So it is three pieces: the predicate, an initialization site that the
# runtime-lib rule allows (or an exemption for those modules), and the read path
# learning the slot. Four attempts, four different walls; recorded so the fifth
# starts past them.
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
