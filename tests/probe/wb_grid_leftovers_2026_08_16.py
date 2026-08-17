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
