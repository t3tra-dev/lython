"""The 2026-08-18 parallel sweep: the 27 findings still live.

Forty agents over ten domains wrote realistic programs, diffed stdout AND
exit code against python3.14, reduced every divergence and named a working
neighbour; an adversarial verifier re-ran both binaries for each. 30
findings survived verification. THREE are already closed (the inline slice
of instances, its __repr__-reads-a-field twin, and the field subscript
store) and are not repeated here. This file is comment-only ON PURPOSE:
every program below FAILS, so none of them can live in a runnable probe. It
runs as an empty program.

Read them as data. Each was reproduced by an agent and by hand for the two
headline entries, but a symptom can MOVE between binaries: finding 8 was
reported as a mis-raised IndexError and showed an ownership error on the
current build, and only bisecting the session's saved binaries showed that
was not a regression. Reproduce before touching. Ordered silent-wrong-
answer, crash, false-refusal, missing-feature -- which is the order to work
them in.
"""

# ==========================================================================
# [SILENT-WRONG-ANSWER] controlflow
# Generator: an int accumulator carried across a `yield` inside a for-loop
# over a list reads its stale pre-loop value, prints a wrong number, then dies
# with "int too large to convert to a native 64-bit integer"
#
#     --- program
#     from typing import Iterator
#     
#     def g(xs: list[int]) -> Iterator[int]:
#         total = 0
#         for x in xs:
#             total = total + x
#             print("inside", total)
#             yield 0
#     
#     for v in g([1, 2]):
#         print(v)
#
# lyc: exit code 1 inside 1 0 inside 2 Traceback (most recent call last): File
# ".../fail.py", line 10, in <module> for v in g([1, 2]): ~^^^^^^^^ File
# ".../fail.py", line 3, in g def g(xs: list[int]) -> Iterator[int]:
# ValueError: int too large to convert to a native 64-bit integer (matches the
# report exactly, including the caret line, which Lython does print here)
# py : exit code 0 inside 1 0 inside 3 0
#
#     --- a neighbour that AGREES
#     from typing import Iterator
#     
#     def g(xs: list[int]) -> Iterator[int]:
#         total = 0
#         for x in xs:
#             total = total + 1
#             print("inside", total)
#             yield 0
#     
#     for v in g([1, 2]):
#         print(v)
#
# verifier: VERDICT: real, and the report understates its scope. Not any
# excluded item — this is a generator codegen miscompile, not a design
# boundary. All files under /private/tmp/claude-501/-Users-user-Desktop-dev-ly
# thon/48b9002d-19db-40fb-8a57-
# f0326e80420f/scratchpad/wf/verify_dom_controlflow_0/ NEIGHBOUR CONFIRMED:
# neigh.py (`total = total + 1`) prints "inside 1 / 0 / inside 2 / 0" on BOTH,
# exit 0 on both. SMALLER REPRODUCER (p1_preinit0.py, 9 lines, no print, no
# arithmetic): from typing import Iterator def g(xs: list[int]) ->
# Iterator[int]: total = 0 for x in xs: total = x yield total for v in g([10,
# 3]): print(v) lyc: prints 10, then ValueError (exit 1). CPython: 10, 3 (exit
# 0). WHAT THE FINDER GO
#
# axes: SILENT WRONG VALUE FIRST: on iteration 2 Lython prints `inside 2`
# where CPython prints `inside 3`, i.e. `total` was re-read as its pre-loop 0
# instead of the carried 1 (`0+2=2`), so a wrong number is printed with no
# diagnostic before the run aborts. The loop variable itself is NOT the
# problem -- `print("x is", x)` prints the correct 1 then 2 (aa3_printx.py).
# Addend axis: `total = total + x` fails, `total = total + 1` agrees
# (cc1_zz4_plus1.py / zz1_plus1.py), and `n = n + 1` agrees (y4_count.py) --
# so it needs the carried value combined with a value read out of the
# container. Iterable axis: `for x in xs` over `list[int]` fails
# (x1_listparam.py), a local list literal fails (x2_listlocal.py), `t

# ==========================================================================
# [SILENT-WRONG-ANSWER] functions
# A method's mutable default argument is re-created on every call instead of
# being shared (silent wrong answer, both exit 0)
#
#     --- program
#     class Bag:
#         def add(self, into: list[int] = []) -> int:
#             into.append(1)
#             return len(into)
#     
#     
#     b = Bag()
#     print(b.add())
#     print(b.add())
#
# lyc: 1 1 [exit 0]
# py : 1 2 [exit 0]
#
#     --- a neighbour that AGREES
#     def add(into: list[int] = []) -> int:
#         into.append(1)
#         return len(into)
#     
#     
#     print(add())
#     print(add())
#     
#     # both compilers print 1 then 2, exit 0
#
# verifier: VERDICT: real, silent wrong answer, both exit 0. The failing
# program and the working neighbour both behave exactly as claimed (lyc 1/1 vs
# cpy 1/2; neighbour 1/2 on both). Extended the neighbour to three calls: lyc
# prints 1/2/3, so the module-scope default genuinely IS shared in Lython —
# the neighbour is not a two-call coincidence. THE FINDER MIS-CHARACTERIZED
# THE AXIS. It is not about methods, `self`, or mutability. The real split is
# `def` inside a class body vs `def` anywhere else. Evidence (all run on both
# binaries): - staticmethod, no `self` at all: diverges (lyc 1/1, cpy 1/2). So
# `self` is irrelevant. - unbound call `Bag.add(b)`: diverges (1/1 vs 1/2). -
# `dict[int,int] = {}` with `into[l
#
# axes: free function vs method: the free function shares the default
# correctly (1,2), only methods diverge. @staticmethod diverges identically
# (1,1). A dict default {} diverges identically (1,1). One instance vs two
# fresh instances (Bag().add() twice): both diverge (1,1). Straight-line vs
# loop: 'for _ in range(3): print(b.add())' gives 1,2,3 in CPython and 1,1,1
# in Lython. A nested function inside a function (def outer(): def
# inner(acc=[])...) AGREES, so it is the class-body def specifically. Passing
# the argument explicitly (b.add(['seed'])) agrees. A default naming a module-
# level list (into: list[int] = STORE) agrees, because the sharing then comes
# from the global rather than from def-time evaluat

# ==========================================================================
# [SILENT-WRONG-ANSWER] functions
# A method's default expression is re-evaluated at call time, so a side-
# effecting default fires an extra time (silent wrong answer, both exit 0)
#
#     --- program
#     def make() -> int:
#         print("eval")
#         return 1
#     
#     
#     class Bag:
#         def add(self, n: int = make()) -> int:
#             return n
#     
#     
#     b = Bag()
#     print(b.add())
#     print(b.add())
#
# lyc: eval 1 eval 1 [exit 0]
# py : eval 1 1 [exit 0]
#
#     --- a neighbour that AGREES
#     def make() -> int:
#         print("eval")
#         return 1
#     
#     
#     def add(n: int = make()) -> int:
#         return n
#     
#     
#     print(add())
#     print(add())
#     
#     # both compilers print eval, 1, 1 and exit 0
#
# verifier: CONFIRMED, byte-for-byte as claimed, and the neighbour agrees
# byte-for-byte on both compilers (eval / 1 / 1, exit 0). Not one of the
# excluded known items. MECHANISM IS MIS-STATED BY THE FINDER. It is not "an
# extra fire on top of the def-time one". Lython does not evaluate a class-
# body def's default at definition time AT ALL; it evaluates it lazily at each
# call site. Two experiments pin this down: - Zero calls
# (/private/tmp/.../wf/verify_dom_functions_1/t13.py): class Bag with `def
# add(self, n: int = make())`, instantiate but never call. Lython prints only
# "never called"; CPython prints "eval" then "never called". Lython evaluates
# it ZERO times where CPython evaluates it once. - Ordering with
#
# axes: Same subsystem as the mutable-default finding above (def-time vs call-
# time evaluation of a method default), but the observable is different and
# worth its own test: the default's side effect is visible in stdout. Note the
# asymmetry Lython shows - 'eval' fires once at class definition AND again on
# the SECOND call, not on the first; so it is not simply 'always re-evaluate'.
# Free function vs method: free function agrees. In a larger program (a class
# method plus a free function both defaulting to make()) CPython printed
# 'eval' twice total and Lython three times.

# ==========================================================================
# [SILENT-WRONG-ANSWER] numbers
# An int literal in a float-annotated container is stored as raw int bits and
# read back as a denormal float — silent wrong answer
#
#     --- program
#     xs: list[float] = [1]
#     print(xs[0])
#
# lyc: 5e-324 (exit 0, empty stderr)
# py : 1 (exit 0, empty stderr)
#
#     --- a neighbour that AGREES
#     xs: list[float] = [1.0]
#     print(xs[0])
#     # both print 1.0, exit 0
#
# verifier: CONFIRMED, ran both binaries myself. fail.py `xs: list[float] =
# [1]` / `print(xs[0])` -> Lython "5e-324" exit 0, CPython "1" exit 0.
# neigh.py with `[1.0]` -> both "1.0" exit 0. Not any excluded item. WHAT THE
# FINDER GOT WRONG — the mechanism claim "stored as raw int bits" is false.
# The bad value does not depend on the int at all: [1] -> 5e-324 [100] ->
# 5e-324 [4607182418800017408] -> 5e-324 (CPython prints 4607182418800017408)
# [-7] -> nan (CPython prints -7) 5e-324 is the double with bit pattern 0x1,
# so what is being read is a fixed small field of the boxed int (limb count /
# sign word), not the value's bits. Positive -> 0x1, negative -> nan. Title
# should say "reads back as a fixed garbage do
#
# axes: literal vs variable: [1] fails, [1.0] agrees — so it is the int
# literal, not the annotation. Container axis: dict[str, float] = {"a": 1} ->
# print(d["a"]) gives 5e-324 vs 1; tuple[float, float] = (1, 2.5) ->
# print(t[0]) gives 5e-324 vs 1; list[float] = [x for x in [1,2,3]] gives
# '5e-324 1.5e-323' vs '1 6' — same corruption through a comprehension. Read-
# path axis: xs[0], sum(xs), max(xs), and 'for v in xs' ALL read the garbage,
# so it is the store not the read. Consumer axis: print(xs) agrees ([0.5, 1,
# 2] both) because printing dispatches on the element's own runtime tag, which
# is why the corruption is invisible until arithmetic. Value axis: list[float]
# = [7, -3] prints '5e-324' then 'nan' (vs

# ==========================================================================
# [SILENT-WRONG-ANSWER] ownership
# Silent wrong answer (both exit 0): a returned/aliased field container
# reports length 0 after the field is rebound, and only because of a LATER
# subscript store
#
#     --- program
#     class Table:
#         def __init__(self) -> None:
#             self.rows: dict[str, int] = {}
#     
#         def rotate(self) -> dict[str, int]:
#             old = self.rows
#             self.rows = {}
#             return old
#     
#     
#     t = Table()
#     t.rows["a"] = 1
#     t.rows["b"] = 2
#     old = t.rotate()
#     print(len(old))
#     old["b"] = 9
#     print(len(old))
#
# lyc: 0 1 exit code 0
# py : 2 2 exit code 0
#
#     --- a neighbour that AGREES
#     class Table:
#         def __init__(self) -> None:
#             self.rows: dict[str, int] = {}
#     
#     
#     def rotate(t: Table) -> dict[str, int]:
#         old = t.rows
#         t.rows = {}
#         return old
#     
#     
#     t = Table()
#     t.rows["a"] = 1
#     t.rows["b"] = 2
#     old = rotate(t)
#     print(len(old))
#     old["b"] = 9
#     print(len(old))
#     
#     # One edit: `rotate` is a free function taking the object instead of a method on it.
#     # CPython: 2 then 2 (rc 0).  Lython: 2 then 2 (rc 0).  AGREE.
#
# verifier: CONFIRMED, and it is worse than reported. I wrote both programs
# myself; failing program gives Lython "0\n1" rc 0 vs CPython "2\n2" rc 0. The
# neighbour gives "2\n2" rc 0 on both — it really does agree. Not any excluded
# item (no dynamic dispatch, no isinstance/object, no starred expr, no union-
# field diagnostic — this is a silent wrong answer with rc 0, and the smallest
# form has no method and no union at all). SMALLER REPRODUCER (9 lines, no
# method, one print). All files under /private/tmp/claude-501/-Users-user-
# Desktop-dev-lython/48b9002d-19db-40fb-8a57-
# f0326e80420f/scratchpad/wf/verify_dom_ownership_1/ (n2.py): class T: def
# __init__(self) -> None: self.d: dict[str, int] = {} t = T() old = t.d
#
# axes: (1) method vs free function: moving `rotate` out of the class to a
# free function `rotate(t: Table)` makes it agree. (2) one statement vs two --
# and this is the striking part: DELETING the trailing `old["b"] = 9` makes
# the FIRST `print(len(old))` print 2 instead of 0. A later statement
# retroactively changes an earlier printed value. (3) store vs method-mutator:
# replacing `old["b"] = 9` with `old.clear()` agrees (2 then 0). (4) accessor
# with vs without the rebind: a `get()` that just does `return self.rows` (no
# `self.rows = {}`) agrees. (5) module scope: inlining the method body at
# module scope (`old = t.rows` / `t.rows = {}`) reproduces identically (0,1 vs
# 2,2), so the method is not required.
#
# ⭐ ONE ROOT WITH THE OTHER OWNERSHIP FINDING, localised 2026-08-18: rebinding a
# field RELEASES the object a live local alias still names. `old = t.rows` binds a
# BORROW, `t.rows = {}` (or `= []`) releases the slot's reference, and the alias is
# then a dangling handle. The neighbours place it exactly:
#
#     old = t.rows; print(len(old))                 -> correct
#     old = t.rows; t.rows = []; print(len(old))    -> correct (nothing decoded yet)
#     old = t.rows; t.rows = []; old[0] = 9         -> IndexError / SIGSEGV
#
# ⛔ JIT AND AOT DISAGREE on the dict spelling, which no sweep column would show:
# `lyc jit` prints 0 then 1 (silent wrong answer) and the AOT binary segfaults, 3/3
# each, from the same source. Measure both when the finding is about lifetime.
#
# ⛔ THREE REPAIRS TRIED 2026-08-18, all reverted, each measured:
#   1. retain the alias at the field read (`retainAggregateSlot`, then bind the
#      bundle as Own) when the read is separated from one of its uses by a store
#      to the same field -- read < store < use in dominance order. The programs
#      became CORRECT in both backends, and every one of them leaked: 2 allocs /
#      8264 B for the list case, 7 / 17094 B for the dict one. Nothing discharges
#      an aggregate-slot retain that has no slot.
#   2. the same plus `markOwnedLocalObjectBundle` on the read's result, so the
#      insertion pass would give it an exit release. No change: still leaking.
#   3. `bindOwnedEvidenceValue` instead, which is how an owned container ELEMENT
#      read is bound. Two neighbouring programs stopped compiling
#      ("ly.ownership.owned_local_object marks a ...") and the leak stayed.
#
# So the retain is the easy half and the discharge is the defect: at a field read
# there is no resource for a frame-lifetime reference to attach to. That is the
# thing to build, and it is why this is a kernel item and not a patch.
#
# ⛔ The sound repair is that a field read of a mutable container OWNS its
# reference rather than borrowing -- a change to the ownership model with a
# retain/release per field read, not a patch. Routing the field's dict stores
# through the runtime payload (the same repair the list store took) was tried and
# REVERTED: it changed nothing measurable on three field-dict programs and turned
# this one's silent wrong answer into a deterministic crash.
#

# ==========================================================================
# [SILENT-WRONG-ANSWER] strings
# A loop-carried accumulator silently resets to its initial value across a
# `yield`, when the generator's for-loop walks a list
#
#     --- program
#     from typing import Iterator
#     def g() -> Iterator[str]:
#         total = 0
#         for n in [5]:
#             total = total + n
#             yield "x"
#         yield f"{total}"
#     for x in g():
#         print(x)
#
# lyc: fail.py: stdout "x\n0\n", exit 0. Reduced min1.py (seen = n, no
# accumulator, no f-string, no trailing yield): stdout "x\n0\n", exit 0.
# min2.py, which reads the local twice in the same resume: "x\ninside after
# resume: 5\nafter loop: 0\n", exit 0. c1.py, initial value 42 instead of 0:
# "x\n0\n", exit 0. Multi-element list (d1.py [5,6], c5.py [1,2,3]): first "x"
# then hard failure, exit 1, stderr trace
# py : fail.py: stdout "x\n5\n", exit 0. min1.py: "x\n5\n", exit 0. min2.py:
# "x\ninside after resume: 5\nafter loop: 5\n", exit 0. c1.py: "x\n5\n", exit
# 0. d1.py: "x\nx\n6\n", exit 0. c5.py: "x\nx\nx\ntotal = 6\n", exit 0.
#
#     --- a neighbour that AGREES
#     from typing import Iterator
#     def g() -> Iterator[str]:
#         total = 0
#         for n in range(5, 6):
#             total = total + n
#             yield "x"
#         yield f"{total}"
#     for x in g():
#         print(x)
#     
#     # both print:  x / 5   exit 0
#
# verifier: CONFIRMED as a silent wrong answer, and the claimed neighbour
# (range(5, 6)) really does agree with CPython (both print x / 5, exit 0). Not
# an excluded item. But the finder's characterisation of the mechanism is
# wrong on two counts, and the title should be rewritten. (1) It does NOT
# "reset to its initial value". c1.py sets seen = 42 before the loop and
# Lython still prints 0, not 42. The read yields zero regardless of the pre-
# loop value. (2) It is NOT a save/restore-across-yield bug. min2.py reads the
# local twice inside a single resume: inside the loop body after the resume it
# reads 5 (correct), and after the loop-exit edge, in the same resume, it
# reads 0. The value survives the suspension int
#
# axes: iterable kind (list literal -> FAILS, range() -> AGREES, tuple literal
# -> also fails); loop vs straight-line (moving `total = total + n` out of the
# loop AGREES); yield inside vs outside the loop body (`for n in [5]: total =
# total + n` with the only yield after the loop AGREES); loop-carried vs loop-
# local (`m = n + 1; yield m` AGREES); number of iterations (`[5]` = silent 0,
# `[5, 6]` = hard ValueError instead); while-loop instead of for (`i = 0;
# while i < 2: ... yield` AGREES). So the trigger is precisely: a name live
# across BOTH the loop back-edge AND a yield, in a for-loop whose iterator is
# a list.

# ==========================================================================
# [SILENT-WRONG-ANSWER] strings
# `int()` applied to a bytes slice is silently a no-op — the bytes object is
# printed instead of the integer
#
#     --- program
#     b = b"0012"
#     print(int(b[0:4]))
#
# lyc: stdout: b'0012' exit code 0
# py : stdout: 12 exit code 0
#
#     --- a neighbour that AGREES
#     s = "0012"
#     print(int(s[0:4]))
#     
#     # both print:  12   exit 0
#
# verifier: CONFIRMED, exactly as reported. Both programs run by me; the
# failing program prints b'0012' (exit 0) under lyc and 12 (exit 0) under
# python3.14. The claimed working neighbour (str slice) really agrees: 12 /
# 12, exit 0 both. Not on the excluded list. SMALLER REPRODUCER (one line, no
# bound name at all): print(int(b"0012"[0:4])) lyc: b'0012' exit 0 | py: 12
# exit 0 THE FINDER UNDERSTATED HOW NARROW THE HOLE IS. `int()` on bytes is
# normally a LOUD error; the slice subscript is the only silent hole I found.
# All of these correctly refuse with `emit error: static type
# !py.contract<"builtins.int"> does not provide manifest method '__init__'`,
# exit 1: print(int(b"12")) # literal, inline b = b"12"; pri
#
# axes: receiver type (bytes slice -> silently identity; str slice ->
# correct); bind-to-a-name vs inline (`c = b[0:4]; print(int(c))` -> honest
# emit error "builtins.int does not provide manifest method '__init__'", so
# only the INLINE slice slips through); literal vs variable
# (`int(b"0012"[0:4])` also prints b'0012'); slice bounds (`int(b[2:4])`
# prints b'12'); downstream use (`int(b[0:4]) + 1` and `int(b[0:4]) == 12` do
# not print a wrong value but die in lowering with "cannot adapt builtins.int
# to runtime input 1 of builtins.bytes.__add__" / `bytes.__eq__`, confirming
# the value kept its bytes type); other converters (`float(b[0:4])` dies with
# "runtime manifest has no builtins.bytes.__float__ method";

# ==========================================================================
# [CRASH] classes
# SIGSEGV when a field initialised from a class attribute is read by __repr__
# and the instance is printed inside a container
#
#     --- program
#     class Account:
#         next_id = 1
#     
#         def __init__(self) -> None:
#             self.ident = Account.next_id
#     
#         def __repr__(self) -> str:
#             return str(self.ident)
#     
#     
#     print([Account()])
#
# lyc: exit code 139 (SIGSEGV). No stdout, no stderr, no diagnostic, no LLVM
# stack trace. Deterministic: 3/3 runs gave 139. Compilation succeeds — the
# crash is at runtime, inside __repr__.
# py : stdout "[1]\n", exit code 0. Matches the claim exactly.
#
#     --- a neighbour that AGREES
#     class Account:
#         next_id = 1
#     
#         def __init__(self) -> None:
#             self.ident: int = Account.next_id
#     
#         def __repr__(self) -> str:
#             return str(self.ident)
#     
#     
#     print([Account()])
#     
#     # adding ": int" to the field: both print "[1]", exit 0
#
# verifier: CONFIRMED. Both programs written out fresh and run by me. failing
# = 139/empty, neighbour = "[1]"/exit 0 in BOTH compilers. Not any excluded
# item. The finder got nothing wrong, but mis-localised the cause: the class
# attribute is incidental. SMALLER REPRODUCER (no class attribute at all, 9
# lines) — /private/tmp/claude-501/-Users-user-Desktop-dev-lython/48b9002d-
# 19db-40fb-8a57-f0326e80420f/scratchpad/wf/verify_dom_classes_2/min_crash.py:
# class A: def __init__(self) -> None: self.x: object = 1 def __repr__(self)
# -> str: return str(self.x) print([A()]) lyc: 139, silent. CPython: "[1]".
# Same 139 with `return repr(self.x)`. REAL TRIGGER: an `object`-typed value
# read inside __repr__ when __repr__ is
#
# axes: annotated vs unannotated field: adding `: int` to the assignment fixes
# it completely. This is the localising edit. | literal vs class-attribute
# initialiser: `self.ident = 7` is fine; `self.ident = Account.next_id`,
# `self.ident = self.next_id`, `self.ident = Account.next_id + 0`, and
# `self.ident = NEXT_ID` (a module global) all crash. Annotating the CLASS
# attribute (`next_id: int = 1`) does not help; only annotating the field
# does. | container vs bare: `a = Account(); print(a)` is CORRECT (prints
# "Account#1"). The crash needs the instance inside a container:
# `print([Account()])`, `xs = [Account()]; print(xs)`, `print((Account(),))`,
# `print({"k": Account()})`, and `print(bank.accounts[0])` whe

# ==========================================================================
# [CRASH] iteration
# `yield from` any operand that is not an inline list literal fails to lower
# (internal "single lane" error)
#
#     --- program
#     from typing import Iterator
#     
#     
#     def relay() -> Iterator[int]:
#         yield from range(2)
#     
#     
#     for v in relay():
#         print(v)
#
# lyc: exit code 1; stdout empty; stderr (verbatim, from
# /private/tmp/claude-501/-Users-user-Desktop-dev-lython/48b9002d-19db-40fb-
# 8a57-f0326e80420f/scratchpad/wf/verify_dom_iteration_0/fail.py):
# loc(fused<{ly.source.end_col = 16 : i32, ly.source.end_line = 8 : i32,
# ly.source.start_col = 9 : i32, ly.source.start_line = 8 :
# i32}>[".../fail.py":8:9]): error: source generator next lowering currently
# support
# py : exit code 0; stdout: 0 1 (stderr empty)
#
#     --- a neighbour that AGREES
#     from typing import Iterator
#     
#     
#     def relay() -> Iterator[int]:
#         yield from [0, 1]
#     
#     
#     for v in relay():
#         print(v)
#
# verifier: VERDICT: real divergence, reproduced byte-for-byte, and the
# neighbour really agrees (`yield from [0, 1]` prints 0/1 on BOTH, exit 0 on
# both). Not any of the excluded items. THE TITLE OVER-CLAIMS. "any operand
# that is not an inline list literal" is wrong on two counts — I found two
# more working shapes: WORKS: `yield from [0, 1]` (list literal), `yield from
# (0, 1)` (TUPLE literal), `yield from [a, b]` / `[a+1, b+1]` (literals with
# variable/computed elements), `yield from inner()` (a call to another
# generator). FAILS (all with the same "single lane" error): `range(2)`; a
# list variable `xs: list[int] = [0,1]; yield from xs`; a tuple variable `t =
# (0,1); yield from t`; a str literal `yield from "
#
# axes: literal vs variable: `yield from [0, 1]` and `yield from [1, 2]` WORK;
# `xs = [1, 2]` then `yield from xs` FAILS; `yield from range(2)`, `yield from
# "ab"` FAIL -- so the operand must be a list literal written at the yield-
# from site. bind-to-a-name vs inline: binding the list to a local is enough
# to break it, so this is not about function boundaries. parameter vs local:
# `yield from xs` with `xs: list[int]` parameter and with `xs: str` parameter
# both FAIL, same error with '!py.contract<"builtins.str">' has 2. one
# statement vs two: rewriting as `for x in xs: yield x` WORKS, which pins the
# defect on the yield-from lowering and not on reading the variable. element
# type: int gives "has 3", str give

# ==========================================================================
# [CRASH] numbers
# Ownership verifier internal error when a local bound to one parameter is
# overwritten by another parameter inside a loop (Euclid gcd)
#
#     --- program
#     def f(a: int, b: int) -> int:
#         x = a
#         while False:
#             x = b
#         return x
#     print(f(1, 2))
#
# lyc: loc(fused<{ly.source.end_col = 13 : i32, ly.source.end_line = 4 : i32,
# ly.source.start_col = 4 : i32, ly.source.start_line = 3 :
# i32}>[".../verify_dom_numbers_1/fail.py":3:4]): error: borrowed entry
# argument 0 of @f is released or transferred without a prior retain Failed to
# run lowering pipeline (exit 1)
# py : 1 (exit 0)
#
#     --- a neighbour that AGREES
#     def f(a: int, b: int) -> int:
#         x = a
#         if False:
#             x = b
#         return x
#     print(f(1, 2))
#     # both print 1, exit 0 — changing only 'while' to 'if' fixes it
#
# verifier: VERDICT: real divergence, reproduced byte-for-byte. The claimed
# diagnostic, both exit codes, and the working neighbour (`while` -> `if`,
# both print 1 / exit 0) are all exactly as reported. Not on the excluded
# list: the only ownership item excluded is "returning a narrowed union FIELD
# from a method (owned resource ... reaches function exit without release)",
# which is a different diagnostic and a different shape — here there is no
# union, no field, and no method required. The finder got four things wrong,
# and the correct characterization is much broader. 1) KIND IS NOT "crash",
# and it is not an internal error. It is a clean exit-1 compile-time rejection
# from the affine-ownership verifier — a SP
#
# axes: Found from a realistic gcd: 'def gcd(a,b): x=a; y=b; while y!=0: t=y;
# y=x%y; x=t; return x' fails with 'borrowed entry argument 1 of @gcd is
# returned as owned without a dominating retain'. loop vs straight-line: 'if'
# agrees, 'while'/'for _ in range(b)' both fail — it is the loop-carried
# merge. literal vs parameter: 'x = 1' in the loop body agrees, 'x = b' fails
# — the RHS must be another borrowed parameter. Pre-loop binding: 'x = 0'
# before the loop agrees, 'x = a' before it fails — the local must already
# alias a parameter. Return axis: replacing 'return x' with 'return 0' still
# fails, so the return is not the trigger, only the reporting site. Type axis:
# identical failure for (a: str, b: str),
#
# ⭐ BIGGER THAN REPORTED, and localised 2026-08-18: EUCLID'S ALGORITHM is
# refused by this, not just the `while False` reduction.
#
#     def gcd(a: int, b: int) -> int:
#         while b != 0:
#             t = b; b = a % b; a = t
#         return a
#     # borrowed entry argument 1 of @gcd is returned as owned without a
#     # dominating retain
#
# The boundary, gridded: `return a` compiles, `x = a; return x` compiles,
# `x = a; if b > 0: x = b; return x` compiles -- a merge has no back edge to
# release on -- and `x = a; while ...: x = b; return x` does not. `x = 0; while
# ...: x = b; return x` compiles too, so the trigger is a PARAMETER as the
# carried local's initial value.
#
# The IR names the imbalance:
#
#     Ly_IncRef(%arg0)              // only after the first repair below
#     cf.br ^bb1(%arg0, %arg1)
#   ^bb1(%0, %1):
#     cond_br ..., ^bb2, ^bb3
#   ^bb2:
#     LyUnicode_DecRef(%0)          // the edge releases the carried value
#     cf.br ^bb1(%arg2, %arg3)      // and forwards the OTHER parameter, no retain
#
# ⛔ TWO REPAIRS TRIED, both reverted:
#   1. acquire the loop's token on the ENTRY edge when the initial carried value
#      is a parameter -- the same thing `acquireUnionCarriedTokens` does for a
#      union, whose note says every other type is the ownership pass's job. It is
#      not: that pass seeds only from OWNED groups and a parameter is a borrow.
#      The retain appears in the IR and the programs stay refused.
#   2. plus: make a parameter bring NO token to a loop lane in
#      `carriedLoopEdgeOperands`'s acquire ledger, which currently reads any
#      block argument as bringing one ("a block argument's token belongs to
#      whichever lane already owns it"). That is wrong for a parameter -- the
#      token belongs to the CALLER -- but changing it alone REGRESSED
#      `x = 0; while False: x = b; return x` from working to refused.
#
# So the ledger and the verifier disagree about who owns a parameter inside a
# loop, and closing it means stating that rule in one place rather than patching
# either end.
#

# ==========================================================================
# [CRASH] numbers
# A float-annotated class attribute with an int default is dropped, then
# raises RuntimeError 'referenced before assignment' at runtime
#
#     --- program
#     class P:
#         v: float = 1
#     print(P.v)
#
# lyc: Traceback (most recent call last): RuntimeError: module global 'P.v'
# referenced before assignment (exit 1)
# py : 1 (exit 0)
#
#     --- a neighbour that AGREES
#     class P:
#         v: float = 1.0
#     print(P.v)
#     # both print 1.0, exit 0
#
# verifier: CONFIRMED verbatim, deterministic over 3 runs. Reported text, both
# exit codes, and the working neighbour (`v: float = 1.0` -> both print 1.0,
# exit 0) all match exactly. Not any excluded item: the class DOES declare the
# field, so this is not "class X has no field", and no dynamic dispatch /
# isinstance / starred / closure / narrowing is involved. The finder got the
# case right but badly undersold it. The real defect is that the class-body
# static-attribute path performs NO annotation/value type agreement check at
# all, while the plain module-global path performs exactly that check.
# Localization (one edit apart, same mismatch): MODULE SCOPE `v: float = 1` +
# print(v) -> clean emit error: "module gl
#
# axes: literal axis: 'v: float = 1.0' agrees, 'v: float = 1' fails — the int
# literal alone. annotation axis: 'v: int = 1' agrees and unannotated 'v = 1'
# agrees, so it needs the float annotation AND the int literal together.
# access axis: reaching it through an instance ('p = P(); print(p.v)') behaves
# the same. Same int-where-float-declared root as the list[float] finding, but
# here the initializer is silently discarded and the failure is reported as a
# bogus 'referenced before assignment' rather than a type error — the
# diagnostic names the wrong cause.

# ==========================================================================
# [CRASH] ownership
# SIGSEGV: alias a field's container into a local, rebind the field, then
# subscript-store through the alias
#
#     --- program
#     class Table:
#         def __init__(self) -> None:
#             self.rows: list[int] = [1, 2]
#     
#     
#     t = Table()
#     old = t.rows
#     t.rows = []
#     old[0] = 9
#     print(len(old))
#
# lyc: No stdout at all; Segmentation fault; exit code 139. Deterministic 3/3
# runs. The AOT path diverges identically: `lyc fail.py -o fail_aot` builds
# cleanly (rc 0) and the binary exits 139, so this is not JIT-only.
# `--release` also crashes.
# py : 2 (exit code 0)
#
#     --- a neighbour that AGREES
#     class Table:
#         def __init__(self) -> None:
#             self.rows: list[int] = [1, 2]
#     
#     
#     t = Table()
#     old = t.rows
#     old[0] = 9
#     print(len(old))
#     
#     # One edit: the `t.rows = []` line is deleted.
#     # CPython: 2 (rc 0).  Lython: 2 (rc 0).  AGREE.
#
# verifier: CONFIRMED, and the finder understated it. Everything below was run
# by me on both binaries in /private/tmp/claude-501/-Users-user-Desktop-dev-ly
# thon/48b9002d-19db-40fb-8a57-
# f0326e80420f/scratchpad/wf/verify_dom_ownership_0. SMALLER REPRODUCER
# (m1.py, 8 lines, no print, no len, single-element list) -> Lython rc 139,
# CPython rc 0: class T: def __init__(self) -> None: self.xs: list[int] = [1]
# t = T() a = t.xs t.xs = [2] a[0] = 3 MECHANISM: storing to an attribute
# destroys the container the attribute previously held, ignoring the live
# alias read out of it. The aliased list's header afterwards reads back length
# 0 or freed memory. m2.py (markers with sys.stdout.flush around the store)
# prints "befor
#
# axes: (1) one statement vs two: deleting the `t.rows = []` rebind makes it
# agree -> the rebind of the field is required. (2) store vs method-mutator:
# `old.append(9)` instead of `old[0] = 9` agrees (prints 3/3) -> only the
# SUBSCRIPT store is broken. (3) store vs read: `print(old[0])` instead of
# `old[0] = 9` agrees (prints 1, 2) -> only the store side. (4) statement
# order: moving `t.rows = []` AFTER the store stops the segv but still
# diverges silently (Lython 2,0 vs CPython 2,2). (5) module scope vs inside a
# function: the same body inside `def go(t: Table) -> int` does not segv, it
# raises a spurious `IndexError: list assignment index out of range` (rc 1) ->
# same wrong container, different symptom. (
#
# ⭐ ONE ROOT WITH THE OTHER OWNERSHIP FINDING, localised 2026-08-18: rebinding a
# field RELEASES the object a live local alias still names. `old = t.rows` binds a
# BORROW, `t.rows = {}` (or `= []`) releases the slot's reference, and the alias is
# then a dangling handle. The neighbours place it exactly:
#
#     old = t.rows; print(len(old))                 -> correct
#     old = t.rows; t.rows = []; print(len(old))    -> correct (nothing decoded yet)
#     old = t.rows; t.rows = []; old[0] = 9         -> IndexError / SIGSEGV
#
# ⛔ JIT AND AOT DISAGREE on the dict spelling, which no sweep column would show:
# `lyc jit` prints 0 then 1 (silent wrong answer) and the AOT binary segfaults, 3/3
# each, from the same source. Measure both when the finding is about lifetime.
#
# ⛔ The sound repair is that a field read of a mutable container OWNS its
# reference rather than borrowing -- a change to the ownership model with a
# retain/release per field read, not a patch. Routing the field's dict stores
# through the runtime payload (the same repair the list store took) was tried and
# REVERTED: it changed nothing measurable on three field-dict programs and turned
# this one's silent wrong answer into a deterministic crash.
#

# ==========================================================================
# [CRASH] stdlib
# `from os import path` makes the compiler fail inside its own posixpath.py
# with a raw MLIR location dump
#
#     --- program
#     from os import path
#     
#     print(path.basename("a/b.py"))
#
# lyc: loc(fused<{ly.source.end_col = 22 : i32, ly.source.end_line = 221 :
# i32, ly.source.start_col = 12 : i32, ly.source.start_line = 221 :
# i32}>["/private/tmp/.../verify_dom_stdlib_0/<stdlib>/posixpath.py":221:12]):
# error: unresolved runtime binding 'path.split' Failed to run lowering
# pipeline [exit=1] (reproduced byte-for-byte, only the absolute path prefix
# differs)
# py : b.py [exit=0]
#
#     --- a neighbour that AGREES
#     import posixpath
#     
#     print(posixpath.basename("a/b.py"))
#
# verifier: VERDICT: the divergence is REAL and reproduced exactly as claimed;
# the neighbour really agrees (`import posixpath` / `posixpath.basename`
# prints `b.py`, exit 0, on both). Not a known-boundary item: `os.path` is
# clearly meant to work, since `import os` + `os.path.basename(...)` prints
# `b.py` exit 0 on both. BUT THE FINDER'S ROOT CAUSE IS WRONG. It has nothing
# to do with `os`, with `from`-imports, or with posixpath.py being "the
# compiler's own file". The real rule is a name-resolution bug in ordinary
# user code: **A name bound to a module M wins over a same-named
# local/parameter for attribute access, whenever the attribute name is also a
# member of M.** Evidence that it is not about `os`: `impor
#
# axes: import-form axis: `import posixpath` works, `from os import getcwd`
# works, `import os` + `os.path.basename` works -> only the name `path`
# entering the module symbol table breaks it. Module-scope-binding axis:
# `import os` plus a user variable `path = "hello"` does NOT break it, and a
# function parameter named `path` does NOT break it -> the capture is specific
# to the imported binding. Failure-site axis: the error is reported at
# posixpath.py:221 (`comps = path.split("/")` inside normpath), i.e. the
# user's import re-binds the *parameter* `path` inside the shipped stdlib
# module, exactly the hazard os.py's own docstring predicts for exporting
# `path`. There is no source-level diagnostic: the compil

# ==========================================================================
# [CRASH] stdlib
# itertools.accumulate with the builtin max/min as the binary function
# explodes the ownership CFG explorer (20000-state abort)
#
#     --- program
#     import itertools
#     
#     last: int = 0
#     for v in itertools.accumulate([3, 1], max):
#         last = v
#     print(last)
#
# lyc: Exact stderr for the reported failing program (fail.py), reproduced
# byte-for-byte, deterministic across 2 runs (retained=1333 both times):
# loc(fused<{ly.source.end_col = 36 : i32, ly.source.end_line = 4 : i32,
# ly.source.start_col = 30 : i32, ly.source.start_line = 4 :
# i32}>[".../fail.py":4:30]): error: ownership CFG exploration exceeded 20000
# states (last: retained=1333 parked=0 borrowed=0 prev=2
# py : 3 exit=0
#
#     --- a neighbour that AGREES
#     import itertools
#     
#     last: int = 0
#     for v in itertools.accumulate([3, 1]):
#         last = v
#     print(last)
#
# verifier: CONFIRMED. Both programs written by me from scratch and run on
# both binaries. The claimed neighbour (drop the func arg) really works and
# agrees: `accumulate([3, 1])` prints 4 under both, exit 0. NOT a known-
# excluded item, and NOT a design boundary — the repo says so itself.
# src/lython/verifier/runtime/AffineOwnership.cpp:2380-2395, immediately above
# the emitError that produced this text: "⚠️ THIS EXIT IS NOT A SAFE-SIDE
# FAILURE, and reading it as one cost a day (2026-07-28) ... The state
# explosion was a cover, not a diagnostic. So a rise in this diagnostic must
# be investigated as a possible masked finding". itertools is not an
# unimplemented module (accumulate demonstrably works), and this is
#
# axes: Function-argument axis: `accumulate(xs)` (default add) works;
# `accumulate(xs, hi)` with `def hi(a: int, b: int) -> int` works; `max` AND
# `min` both blow up -> the builtin as the combinator is the trigger. Loop-
# body axis: body `print(v)` alone works, body `last = v` or `out.append(v)`
# aborts -> the value must be stored. Scope axis: module scope and inside a
# function both abort identically. Length axis: 2-element and 3-element
# sources both abort (state count is identical at 1333, so it is not input-
# size driven). This is what a realistic `running_max` helper compiles to.

# ==========================================================================
# [CRASH] strings
# Generator resumption destroys a loop-carried int: second `next()` raises
# "int too large to convert to a native 64-bit integer"
#
#     --- program
#     from typing import Iterator
#     def g() -> Iterator[int]:
#         total = 0
#         for n in [1, 2]:
#             total = total + n
#             yield total
#     for x in g():
#         print(x)
#
# lyc: Failing program (f03_bad.py) — deterministic across 3 runs, exit code
# 1: stdout: 1 stderr: Traceback (most recent call last): File
# ".../f03_bad.py", line 7, in <module> for x in g(): ~^^ File
# ".../f03_bad.py", line 2, in g def g() -> Iterator[int]: ValueError: int too
# large to convert to a native 64-bit integer (lyc prints the absolute path,
# not the bare "f03_bad.py" the finder quoted; and lyc DOE
# py : Failing program: stdout "1\n3\n", stderr empty, exit code 0. Neighbour:
# stdout "1\n3\n", exit code 0.
#
#     --- a neighbour that AGREES
#     from typing import Iterator
#     def g() -> Iterator[int]:
#         total = 0
#         for n in range(1, 3):
#             total = total + n
#             yield total
#     for x in g():
#         print(x)
#     
#     # both print:  1 / 3   exit 0
#
# verifier: VERDICT: real divergence, reproduced verbatim. But the finder's
# localization is WRONG, and the bug is bigger and more valuable than reported
# (it has a silent-wrong-answer form). WHAT THE FINDER GOT WRONG — the axis is
# not "list literal vs range()": The claimed neighbour works, but for an
# irrelevant reason (switching to range() also removes the container element
# that feeds the accumulator). range() is NOT the fix; these all FAIL with the
# identical ValueError while using `for n in range(2)`: - g1.py: `xs=[10,20];
# total=0; for n in range(2): total = total + xs[0]; yield total` - g4.py:
# same with a dict — `d={"k":5}; total = total + d["k"]` - g5.py: same with a
# tuple — `xs=(10,20)` - g6.py: same
#
# axes: iterable kind (list literal -> crash, range() -> AGREES); iteration
# count (`[1]` AGREES — only one resumption happens, so nothing is read back);
# order of update and yield (`yield total` then `total = total + n` crashes
# the same way, so it is not the update site); `total += n` vs `total = total
# + n` (both crash); loop-local only (`m = n + 1; yield m` AGREES); while-loop
# with the same accumulator AGREES; consumption style (`for x in g()` and
# `list(g())` both crash — `list(g())` crashes before printing anything). Same
# root as finding 1: the frame slot for a name live across the back-edge and
# the yield is not restored, and here it comes back as a non-canonical int
# payload.

# ==========================================================================
# [CRASH] unions
# Iterating a list whose element type is a union is refused by the lowering
# pipeline
#
#     --- program
#     xs: list[int | None] = [1, None]
#     for x in xs:
#         print(x)
#
# lyc: $ /Users/user/Desktop/dev/lython/build/bin/lyc jit fail.py
# loc(fused<{ly.source.end_col = 11 : i32, ly.source.end_line = 2 : i32,
# ly.source.start_col = 9 : i32, ly.source.start_line = 2 :
# i32}>["<scratch>/fail.py":2:9]): error: iteration over a runtime-mode list
# of '!py.union<!py.contract<"builtins.int">, !py.literal<None>>' requires
# rank-1 memref physical values, got 'i64' Failed to run lowering
# py : $ python3.14 fail.py 1 None exit=0 For the smaller reproducer (`for x
# in [1, None]: pass`): no output, exit=0.
#
#     --- a neighbour that AGREES
#     xs: list[int] = [1, 2]
#     for x in xs:
#         print(x)
#
# verifier: VERDICT: real divergence, correctly described, not a known-
# excluded item, and reducible much further than reported. Reproduced
# verbatim: the claimed diagnostic text matches character-for-character (only
# the path differs). Lython exit 1 / CPython exit 0 printing "1\nNone".
# Claimed neighbour verified: `xs: list[int] = [1, 2]` + for loop prints
# "1\n2" exit 0 on BOTH binaries. So the neighbour really agrees. NOT a design
# boundary. The excluded list does not cover this, and the compiler plainly
# accepts the type elsewhere: - `xs: list[int | None] = [1, None];
# print(xs[0]); print(xs[1])` -> Lython exit 0, prints "1\nNone", identical to
# CPython. Same declared type, same list, only the ACCESS MODE di
#
# axes: element type (list[int] iterates fine; list[int|None], list[str|None],
# list[int|str] all fail) -> the union is the trigger; read style
# (`print(xs[0])` on the SAME union list works, so the storage is readable,
# only the iterator ABI is wrong); module scope vs inside a function (both
# fail, same message); tuple[int|None, ...] instead of list gives the sibling
# message 'list iteration evidence candidate 1 has a different physical ABI
# shape'.

# ==========================================================================
# [CRASH] unions
# Subscripting a union-element list with a non-constant index is refused
#
#     --- program
#     xs: list[int | None] = [1, None]
#     i: int = 0
#     print(xs[i])
#
# lyc: loc(fused<{ly.source.end_col = 11 : i32, ly.source.end_line = 3 : i32,
# ly.source.start_col = 6 : i32, ly.source.start_line = 3 :
# i32}>["<path>/fail.py":3:6]): error: sequence __getitem__ evidence candidate
# 1 has a different physical ABI shape Failed to run lowering pipeline (exit
# 1)
# py : 1 (exit 0)
#
#     --- a neighbour that AGREES
#     xs: list[int | None] = [1, None]
#     print(xs[0])
#
# verifier: CONFIRMED as a real divergence. Ran both binaries on both programs
# myself. fail.py: lyc exit 1 with the quoted diagnostic, python3.14 prints
# "1" exit 0. neigh.py (print(xs[0])): both print "1" exit 0, so the working
# neighbour is genuine and one edit away. SMALLER REDUCER (2 lines, drops the
# variable binding entirely): xs: list[int | None] = [1, None] print(xs[0 +
# 0]) This fails with the byte-identical diagnostic (only column numbers
# differ). THE FINDER'S CHARACTERIZATION IS WRONG on the trigger. It is not
# "non-constant index" — it is "index is not a bare integer literal".
# Evidence: xs[0] works, xs[-1] works (prints None, both agree), but the
# compile-time constant xs[0 + 0] FAILS, as do xs[i]
#
# axes: literal vs variable index (xs[0] works, xs[i] fails -- this is the
# whole difference); element type (list[int] with xs[i] works); module scope
# vs function (inside a `def` the same code fails with a DIFFERENT message,
# 'runtime manifest has no builtins.list.__getitem__ method'); while-loop
# indexing gives the function-scope message too.

# ==========================================================================
# [CRASH] unions
# Subscripting a union-valued dict with a non-constant key is refused
#
#     --- program
#     d: dict[str, str | None] = {"x": "set", "y": None}
#     k: str = "x"
#     print(d[k])
#
# lyc: Compile-time failure, exit code 1, nothing printed to stdout:
# loc(fused<{ly.source.end_col = 10 : i32, ly.source.end_line = 3 : i32,
# ly.source.start_col = 6 : i32, ly.source.start_line = 3 :
# i32}>["<path>/fail.py":3:6]): error: dict __getitem__ evidence candidate 1
# has a different physical ABI shape Failed to run lowering pipeline
# py : set (exit code 0)
#
#     --- a neighbour that AGREES
#     d: dict[str, str | None] = {"x": "set", "y": None}
#     print(d["x"])
#
# verifier: VERDICT: real divergence, matches the report essentially verbatim
# (including the exact error string and both exit codes). Neighbour genuinely
# agrees with CPython (`print(d["x"])` -> "set", exit 0 on both). Not any of
# the excluded known items. SMALLER REPRODUCER (2 lines, no annotation on the
# dict, no separate binding): k: str = "x" print({"x": "a", "y": None}[k]) ->
# lyc: same "dict __getitem__ evidence candidate 1 has a different physical
# ABI shape", exit 1; py3.14: "a", exit 0. Also fails as a bare statement with
# no print at all (`d[k]` alone), so print/formatting is not involved. WHAT
# THE FINDER GOT WRONG: 1. KIND is mislabelled as "crash". There is no signal
# and no abort — it is a clean c
#
# axes: literal vs variable key (d["x"] works, d[k] fails); value type
# (dict[str,str] with d[k] works; dict[str,int|str] fails the same way); loop
# key (`for key in [...]: d[key]` fails identically, which is how it shows up
# in real code); one statement vs two (inlining the key as a literal is the
# only thing that helps).

# ==========================================================================
# [FALSE-REFUSAL] controlflow
# An `except` handler that ends in `return` is not treated as a terminator, so
# a name bound only in the try body is rejected as unresolved; the same
# handler ending in `raise` is accepted
#
#     --- program
#     def f(s: str) -> int:
#         try:
#             n = int(s)
#         except ValueError:
#             return 0
#         return n
#     print(f("5"))
#
# lyc: exit code 1, stderr: `fail_ret.py:6:11: emit error: unresolved name
# 'n'` (no stdout)
# py : exit code 0, stdout: `5`
#
#     --- a neighbour that AGREES
#     def f(s: str) -> int:
#         try:
#             n = int(s)
#         except ValueError:
#             raise RuntimeError("bad")
#         return n
#     print(f("5"))
#
# verifier: CONFIRMED as a divergence — I wrote both programs myself and ran
# both binaries. Failing program: lyc exit 1 with `unresolved name 'n'` at
# 6:11; python3.14 exit 0 printing `5`. Neighbour (handler ends in `raise
# RuntimeError("bad")`): both exit 0 printing `5`, so the neighbour really
# agrees. Not in the excluded list, and not a documented boundary — the name
# is DEFINITELY bound at the read (the join is dominated by the binding arm),
# so this is a false refusal, not "statically unresolvable". THE FINDER'S
# MECHANISM CLAIM IS WRONG. It is not "return is not treated as a terminator
# while raise is". In a plain if/else, `raise` fails identically: def f(b:
# bool) -> int: if b: n = 1 else: raise RuntimeE
#
# axes: Terminator-kind axis (the localiser): handler ends in `return 0` ->
# rejected; handler ends in `raise RuntimeError("bad")` -> accepted and prints
# 5 (m_raise.py). Both make `n` definitely bound after the try, so definite-
# assignment is only crediting `raise`. Assignment axis: if the handler also
# assigns `n = 0` it is accepted (n1_assign_both.py). Use-site axis: using `n`
# inside the try body instead of after it is accepted (n3_return_inside.py).
# Scope axis: module scope shows the same hole -- a handler that merely falls
# through leaves `n` unresolved (n4_module_return.py), while assigning in both
# arms works (r_tryname.py). Statement-count axis: adding `print("skip")`
# before the `return` changes n
#
# ⭐ LOCALISED 2026-08-18, cause exact. `postTryLanesAvailable` is
# `!hasElse && !hasFinally && !usesFinallyCompletion && handlers`, and a `return`
# ANYWHERE in the statement sets `protectedBodyHasReturn`, which with a non-None
# return type sets `supportsValueReturnThroughFinally` and therefore
# `usesFinallyCompletion`. So the post-try lanes -- the only channel that carries
# a try-body binding to the continuation -- are switched off by the presence of
# the return, and the name reads as unresolved.
#
# That is why the `raise` spelling of the same handler WORKS: `raise` does not set
# `handlerBodyHasReturn`, the lanes stay on, and the lane carries `n`.
#
# ⛔ Publishing the try-end bindings directly is not available: they are SSA
# values inside the try region and do not dominate the continuation, which is why
# the lanes exist at all.
#
# ⭐ THE COEXISTENCE WAS BUILT AND MEASURED 2026-08-19, AND IT IS NOT THE
# RESULT-INDEX ACCOUNTING THAT BLOCKS IT. The emitter side works: lanes appended
# after the completion flags and the return payload, completion yields padded
# with inert lane defaults once the lane set is known, handler-exit yields
# carrying the completion prefix, `postLaneBase` on every read. Measured on eight
# programs -- the motivating idiom, two names, a str lane, a `return` in the try
# body instead, two handlers, a loop around it -- ALL AGREE with CPython. The
# patch is kept at
# /private/tmp/.../scratchpad/try-lane-completion.patch (207 lines).
#
# What stops it is OWNERSHIP, and the blocker is exact. A return path must put an
# inert value in every lane and a fall-through must put one in the payload, so
# each result is owned on one path and dead on the other -- and the continuation
# then reads `decref(flag ? lane : payload)` next to `return (flag ? payload :
# lane)`. When there is exactly ONE lane the two arms are structurally identical,
# py-level canonicalize folds them into six selects over the same pair, and the
# alias analysis (which sees through a select in both directions, by design)
# fuses lane and payload into ONE class with TWO consumes. The unfold rule then
# credits one reference and mints a retain per class: 56 B leaked per successful
# call of the motivating program, 82 B for the str lane, 112 B for two calls.
# Two names, or any continuation that does more before returning, does not merge
# and does not leak -- which is why the leak looks intermittent.
#
# ⛔ The tree's own answer for a select over owned handles -- expand it back into
# the diamond it was folded from (Ownership.cpp, "expand-object-selects") -- does
# not reach it, because `frameProduces` says no to a BLOCK ARGUMENT, and a py.try
# result is exactly that once the regions are inlined. Teaching it the edge walk
# (the one `valueDerivedFromEntryArgument` already does for entry arguments) was
# measured: the diamond then double-releases, because a SWAP forwards both
# objects into two destination groups on both arms, so the arms fuse the two
# groups instead of letting the loser die. "released owned resource from
# @LyLong_FromStr ... more than once" on all five single-lane programs. Reverted.
#
# ⛔ An inert slot that is not owned would end it -- a NULL object handle, which
# py.incref/py.decref already document as safe to receive. The dialect has no
# way to spell one, and inventing it is the round this needs.
#
# ⛔ A user class or a list lane must stay out of the coexistence for a separate
# reason: emitDefaultReturnValue spells no inert value for them and falls back to
# None, which published a None as a Box and aborted in Ly_DecRef ("observed
# non-positive refcount").
#
# ⛔ The storage promotion cannot substitute: it deliberately skips a name NOT
# bound before the try ("the handler cannot observe a value the body may never
# have produced"), and `n` is first bound inside it.
#

# ==========================================================================
# [FALSE-REFUSAL] controlflow
# A for-loop target is not readable after the loop finishes; `for i in ...`
# then `print(i)` is rejected as an unresolved name
#
#     --- program
#     for i in range(3):
#         print(i)
#     print("last", i)
#
# lyc: exit code 1, stdout empty, stderr: fail.py:3:14: emit error: unresolved
# name 'i'
# py : exit code 0, stderr empty, stdout: 0 1 2 last 2
#
#     --- a neighbour that AGREES
#     def g(xs: list[int]) -> int:
#         i = -1
#         for i in range(len(xs)):
#             if xs[i] == 5:
#                 break
#         return i
#     print(g([3, 5, 7]))
#
# verifier: CONFIRMED, but the finder mis-titled the mechanism and missed a
# much smaller reproducer. WHAT I RAN (all files under
# .../scratchpad/wf/verify_dom_controlflow_2/): - fail.py -> LY rc=1
# "unresolved name 'i'" / CP rc=0 "0 1 2 last 2". Byte-for-byte as claimed. -
# neigh.py -> both rc=0, both print "1". The claimed neighbour genuinely
# agrees. THE TITLE IS WRONG. This has nothing to do with for-loop targets, or
# with loops. The real rule is: a name whose FIRST binding is inside a nested
# suite does not exist after that suite. I hit the same "unresolved name"
# diagnostic with no loop target involved: - if-body: `if 1 < 2:\n k =
# 7\nprint(k)` -> unresolved name 'k' - while-body: name first bound in a whi
#
# axes: Pre-binding axis (the localiser): assigning the name once before the
# loop makes the same read work and prints 1 (lv_preinit.py) -- so the loop
# body's binding is discarded at loop exit rather than the name being unknown.
# Scope axis: fails identically at module scope (lv_module.py) and inside a
# function (r_loopvar.py). Loop-kind axis: `while` loops are unaffected
# because their control variable must already be initialised before the test,
# which is why tests/golden/cases/loop_else.py exercises `while i < 3` after
# `i = 0` and never reads a for-target after its loop. Break axis: present
# with and without `break` in the body. This is the idiom `for i in ...: if
# ...: break` / `return i`, and also bre
#
# ⭐ MEASURED 2026-08-18, and the honest part is what it CANNOT be. The loop target
# is bound inside the body's scope and dropped at the end; making it a carried
# local would publish the last iteration's value, which is CPython's answer -- for
# a loop that RAN. `for i in []: pass` then `print(i)` is a NameError in CPython,
# and a static binding cannot express "bound only if the loop ran" without a
# definedness flag, which is the cell-with-a-sentinel feature this tree does not
# have. Binding it unconditionally would print a seed value where CPython raises:
# a silent wrong answer traded for a false refusal.
#
# ⛔ The sound SUBSET is a name already bound before the loop (`i = 0;
# for i in ...`), which the carried-local machinery already handles as an ordinary
# rebind -- and which is not the idiom that motivates the finding.
#

# ==========================================================================
# [FALSE-REFUSAL] dataflow
# dict.setdefault() called inside a loop loses ownership of a locally-built
# dict, falsely rejecting the function
#
#     --- program
#     def f() -> dict[str, int]:
#         d: dict[str, int] = {}
#         for c in "ab":
#             d.setdefault(c, 0)
#         return d
#     
#     
#     print(f())
#
# lyc: exit code 1, stdout empty. stderr (only the trailing text varies with
# source path): loc(fused<{ly.source.end_col = 12 : i32, ly.source.end_line =
# 5 : i32, ly.source.start_col = 4 : i32, ly.source.start_line = 5 :
# i32}>[".../bad.py":5:4]): error: owned resource from @LyDict_FromLength
# result 0 reaches function exit without release, transfer, or owned return
# Failed to run lowering pipeline The locat
# py : exit code 0, stdout: {'a': 0, 'b': 0}
#
#     --- a neighbour that AGREES
#     def f() -> dict[str, int]:
#         d: dict[str, int] = {}
#         for c in "ab":
#             d[c] = 0
#         return d
#     
#     
#     print(f())
#
# verifier: CONFIRMED, and the finder's characterization is too narrow on two
# counts. 1. The loop is NOT the trigger. A plain `if` reproduces it, with no
# iteration at all — this is a strictly smaller reproducer (6 lines,
# /private/tmp/claude-501/-Users-user-Desktop-dev-lython/48b9002d-19db-40fb-
# 8a57-f0326e80420f/scratchpad/wf/verify_dom_dataflow_0/p3_if.py): def f(flag:
# bool) -> dict[str, int]: d: dict[str, int] = {} if flag: d.setdefault("a",
# 0) return d print(f(True)) lyc: exit 1, `... is still owned when a call to
# 'LyDict_Len' may unwind out of the function ...`; CPython: exit 0, {'a': 0}.
# Its one-edit neighbour `d["a"] = 0` (t2_neighbour_of_if.py) prints {'a': 0}
# exit 0 on both. A bare `try:` region
#
# axes: loop vs straight-line: two unrolled `d.setdefault(...)` calls with no
# loop COMPILE AND AGREE ({'a': 0, 'b': 0}); the loop is required to trigger.
# for vs while: a `while i < 2:` loop fails identically, so it is any back
# edge, not the for-iterator. module scope vs inside a function: the same loop
# at module scope AGREES -- only a function-local dict is affected. local vs
# parameter: `def f(src: dict[str,int])` doing `src.setdefault(c,0)` in a loop
# AGREES, so only a dict the function itself allocates loses ownership. method
# choice, same loop: `d[c] = 0` AGREES (working neighbour), `d.get(c, 0)`
# AGREES, `d.pop(c, 0)` AGREES, `d.update(other)` AGREES -- setdefault is the
# only dict method that break
#
# ⭐ LOCALISED 2026-08-18, and two repairs measured and reverted. `setdefault`
# lowers to a conditional insert whose two arms branch to ONE merge block passing
# the same dict, so everything after the merge names the block argument. Inside a
# loop that argument is loop-carried, and the return then hands back a name the
# owned-return check cannot relate to the `LyDict_FromLength` result. Neighbours:
#
#     module scope, no return          -> compiles
#     setdefault with no loop          -> compiles
#     `d[c] = 0` in the same loop      -> compiles   (keeps the original name)
#     setdefault in a loop + return    -> refused
#
#   1. Union every uniform merge argument with its incoming value in
#      `AliasAnalysis::build` -- the natural home for a rename. 36 TESTS DOWN at
#      once: the alias relation feeds release PLACEMENT everywhere, and making two
#      names one moves releases that were correct.
#   2. Resolve the rename only in `returnTransfersGroup`, additively (try the raw
#      operands first, the resolved ones second, so it can only accept MORE).
#      Sound -- the suite stays green -- and it fixes nothing: the loop case's
#      returned name is a LOOP-CARRIED argument, whose incoming values differ by
#      construction, so the resolver correctly declines. Reverted as no-effect
#      surface.
#
# What is left is the group following a loop-carried argument, which the pass
# already does for a self-forwarding continue edge ("neither a use nor a death")
# and not for a value that ENTERS the loop from the preheader.
#

# ==========================================================================
# [FALSE-REFUSAL] dataflow
# Binding a dict view to a name breaks; d.keys()/values()/items() work only
# when consumed inline
#
#     --- program
#     d: dict[str, int] = {"a": 1}
#     ks = d.keys()
#     print(len(ks))
#
# lyc: exit code 1, stdout empty. stderr (verbatim, for the reported 3-line
# failing program): loc(fused<{ly.source.end_col = 13 : i32,
# ly.source.end_line = 2 : i32, ly.source.start_col = 5 : i32,
# ly.source.start_line = 2 :
# i32}>["/private/tmp/.../verify_dom_dataflow_2/fail.py":2:5]): error: runtime
# manifest has no builtins.dict.keys method Failed to run lowering pipeline
# The message tracks the method nam
# py : exit code 0, stdout: 1
#
#     --- a neighbour that AGREES
#     d: dict[str, int] = {"a": 1}
#     print(len(d.keys()))
#
# verifier: CONFIRMED REAL. I wrote both programs myself and ran both
# binaries; the failing program's stderr and exit code match the claim
# character-for-character (including the 2:5 location), and the claimed
# neighbour `print(len(d.keys()))` really works (Lython exit 0, stdout "1",
# identical to CPython). NOT a known/excluded item, and not a design boundary.
# `src/lython/runtime/modules/builtins.mlir:710` lists "keys", "values",
# "items" in `method_names` for `builtins.dict` with full `method_contracts`,
# and lines 787+ declare `py.class @dict_keys` / `@dict_values` /
# `@dict_items` with variance and base args. So the type layer promises these
# methods exist (which is why the emitter accepts the program) and
#
# axes: bind-to-a-name vs use-inline (the decisive axis): every inline use
# AGREES -- `len(d.keys())`, `list(d.keys())`, `sorted(d.keys())`, `for k in
# d.keys():`, `"a" in d.keys()`, and `for k, v in d.items():` all compile and
# match CPython. Assigning the view to a local first fails. which view: all
# three break the same way -- `ks = d.keys()` -> 'no builtins.dict.keys
# method', `vs = d.values()` -> 'no builtins.dict.values method', `its =
# d.items()` -> 'no builtins.dict.items method'. consumer after the bind:
# `len(ks)`, `list(ks)`, and `for k in ks:` all fail, so the failure is at the
# bind, not the consumer. module scope vs inside a function: `def f(d): ks =
# d.keys(); return len(ks)` fails identically

# ==========================================================================
# [FALSE-REFUSAL] iteration
# A generator object passed as a function parameter cannot be iterated at all
#
#     --- program
#     from typing import Iterator
#     
#     
#     def src() -> Iterator[int]:
#         yield 1
#         yield 2
#     
#     
#     def consume(it: Iterator[int]) -> int:
#         total = 0
#         for s in it:
#             total = total + s
#         return total
#     
#     
#     print(consume(src()))
#
# lyc: exit code 1; stdout empty; stderr: loc(fused<{ly.source.end_col = 15 :
# i32, ly.source.end_line = 11 : i32, ly.source.start_col = 13 : i32,
# ly.source.start_line = 11 : i32}>[".../f1_bad.py":11:13]): error: a
# generator returned out of a function cannot be resumed: the frame it resumes
# into is not carried by the returned value. Call the generator in the for
# statement, bind it to a local in the same f
# py : exit code 0; stdout: 3
#
#     --- a neighbour that AGREES
#     from typing import Iterator
#     
#     
#     def src() -> Iterator[int]:
#         yield 1
#         yield 2
#     
#     
#     def consume() -> int:
#         total = 0
#         for s in src():
#             total = total + s
#         return total
#     
#     
#     print(consume())
#
# verifier: CONFIRMED, byte-for-byte as claimed. I wrote both programs myself.
# f1_bad.py: Lython exit 1 with that diagnostic, CPython exit 0 printing 3.
# f2_neighbour.py: both exit 0, both print 3. The neighbour genuinely agrees.
# WHAT THE FINDER GOT WRONG — the title is wrong about the cause, and the
# defect is broader and worse than reported. 1) IT IS NOT ABOUT GENERATORS.
# The same "a generator returned out of a function cannot be resumed" fires
# for a program that contains no generator and no `yield` anywhere. Smaller
# reproducer (8 lines, /private/tmp/claude-501/-Users-user-Desktop-dev-lython/
# 48b9002d-19db-40fb-8a57-
# f0326e80420f/scratchpad/wf/verify_dom_iteration_1/r1_min.py): from typing
# import Iterator
#
# axes: parameter vs same-function call: calling `src()` inside consume's own
# for statement WORKS; receiving the same object as a parameter FAILS. bind-
# to-a-name: `g = src()` in the caller then `consume(g)` FAILS the same way.
# what the callee does with it: `for s in it` FAILS, `sum(it)` FAILS with the
# same text, `next(it)` FAILS with a different one ("runtime manifest has no
# types.GeneratorType.__next__ method"). annotation: changing the parameter to
# `list[int]` and passing `list(src())` WORKS. return type of the callee:
# returning `int` or building a `list` both FAIL, so it is the incoming value,
# not the result. The diagnostic's own advice ("bind it to a local in the same
# function") does not apply -

# ==========================================================================
# [FALSE-REFUSAL] iteration
# Inside a generator, `for x in other_gen(): yield x` is refused by a
# diagnostic whose advice the program already follows
#
#     --- program
#     from typing import Iterator
#     
#     
#     def src() -> Iterator[int]:
#         yield 1
#         yield 2
#     
#     
#     def relay() -> Iterator[int]:
#         for s in src():
#             yield s
#     
#     
#     for v in relay():
#         print(v)
#
# lyc: exit code 1; stdout empty; stderr: loc(fused<{ly.source.end_col = 18 :
# i32, ly.source.end_line = 10 : i32, ly.source.start_col = 13 : i32,
# ly.source.start_line = 10 : i32}>[".../fail.py":10:13]): error: a generator
# returned out of a function cannot be resumed: the frame it resumes into is
# not carried by the returned value. Call the generator in the for statement,
# bind it to a local in the same fun
# py : exit code 0; stdout: 1 2
#
#     --- a neighbour that AGREES
#     from typing import Iterator
#     
#     
#     def src() -> Iterator[int]:
#         yield 1
#         yield 2
#     
#     
#     def relay() -> Iterator[int]:
#         yield from src()
#     
#     
#     for v in relay():
#         print(v)
#
# verifier: CONFIRMED, and the finder understated it. All files under
# /private/tmp/claude-501/-Users-user-Desktop-dev-lython/48b9002d-19db-40fb-
# 8a57-f0326e80420f/scratchpad/wf/verify_dom_iteration_2/ The claimed
# neighbour really agrees: neighbour.py (`yield from src()`) is lyc exit=0
# stdout "1\n2", identical to CPython. THE REAL TRIGGER IS BROADER THAN "for".
# The refusal fires whenever a generator object is consumed by anything other
# than `yield from` *inside a function that is itself a generator*. Same
# message, all of these: - fail.py `for s in src(): yield s` -> refused -
# d_no_yield_in_loop.py `for s in src(): total += s` then `yield total` (inner
# generator fully drained BEFORE the outer's only yield)
#
# axes: The diagnostic says "Call the generator in the for statement" and the
# program does exactly that, so the message is self-contradicting. inside a
# generator vs inside a plain function: the identical `for s in src():` loop
# in a NON-generator function WORKS (`def consume() -> int: for s in src():
# total += s`), so the trigger is the enclosing `yield`. one statement vs two:
# `yield from src()` WORKS. bind-to-a-name: `g = src()` then `for s in g:
# yield s` FAILS with the same text. loop body: replacing `yield s` with
# `print(s)` and yielding a constant afterwards still FAILS, so it is the
# coexistence of the consumed generator and any yield in the same body, not
# the yielded value.

# ==========================================================================
# [FALSE-REFUSAL] stdlib
# A method's `X | None` parameter called with the literal None loses `is not
# None` narrowing; Counter.most_common(None) fails inside the shipped
# collections.py
#
#     --- program
#     import collections
#     
#     c: collections.Counter = collections.Counter(["a", "b", "a"])
#     print(c.most_common(None))
#
# lyc: /private/tmp/.../wf/verify_dom_stdlib_2/<stdlib>/collections.py:183:15:
# emit error: static type !py.literal<None> does not provide manifest method
# '__lt__'
# /private/tmp/.../wf/verify_dom_stdlib_2/<stdlib>/collections.py:188:14: emit
# error: static type !py.contract<"builtins.int"> does not provide manifest
# method '__lt__' [exit=1]
# py : [('a', 2), ('b', 1)] [exit=0]
#
#     --- a neighbour that AGREES
#     import collections
#     
#     c: collections.Counter = collections.Counter(["a", "b", "a"])
#     print(c.most_common())
#
# verifier: VERDICT: real false-refusal, reproduced byte-for-byte (both
# claimed diagnostics, both line:col) and the neighbour `c.most_common()`
# really agrees ([('a', 2), ('b', 1)], exit 0 on both). Not on the excluded
# list: collections.Counter is not among the unimplemented modules
# (re/deque/namedtuple) and Counter itself works; this is a manifest-method
# refusal on a PARAMETER, not the excluded ownership error on a returned
# narrowed union FIELD. BUT THE TITLE MIS-ATTRIBUTES THE CAUSE. It is not
# about methods, not about `X | None`, and not about parameters. Much smaller
# reproducer, 4 lines, module scope, no class/function/import
# (p1_module_none.py): n = None if n is not None: print(n + 1) print("ok") Lyt
#
# axes: Argument-form axis: `most_common()` works, `most_common(2)` works, and
# binding the None first (`n: int | None = None; c.most_common(n)`) works --
# only the inline None LITERAL fails. Callable-kind axis: reduced out of the
# stdlib into 11 lines of user code -- `class Box: def cap(self, n: int | None
# = None) -> int: limit: int = 3; if n is not None: (if n < limit: limit = n);
# return limit` then `b.cap(None)` gives `emit error: static type
# !py.literal<None> does not provide manifest method '__lt__'`, while the
# IDENTICAL body as a free function `def cap(n: int | None = None)` called
# `cap(None)` compiles and prints 3. So it is method-specific, not narrowing-
# in-general. Guard-shape axis: rewriting t

# ==========================================================================
# [MISSING-FEATURE] dataflow
# list.extend() accepts only a list; a tuple, set, dict, str, or range
# argument is rejected with a raw memref type error
#
#     --- program
#     xs: list[int] = []
#     xs.extend((1, 2))
#     print(xs)
#
# lyc: exit code 1, stdout empty. stderr: loc(fused<{ly.source.end_col = 17 :
# i32, ly.source.end_line = 2 : i32, ly.source.start_col = 0 : i32,
# ly.source.start_line = 2 : i32}>["<dir>/F_tuple.py":2:0]): error: cannot
# adapt builtins.tuple to runtime input 1 of builtins.list.extend [values:
# 'memref<14xi64>', expected 'memref<9xi64>'] Failed to run lowering pipeline
# py : exit code 0, stdout: [1, 2]
#
#     --- a neighbour that AGREES
#     xs: list[int] = []
#     xs.extend([1, 2])
#     print(xs)
#
# verifier: Fully reproduced verbatim on both binaries; the claimed neighbour
# xs.extend([1, 2]) prints [1, 2] / exit 0 on both. Not an excluded item.
# SMALLER REPRODUCER (one statement, 13 chars, no annotations, no variables):
# [].extend(()) Lython: exit 1, same "cannot adapt builtins.tuple to runtime
# input 1 of builtins.list.extend [values: 'memref<14xi64>', expected
# 'memref<9xi64>']" + "Failed to run lowering pipeline". CPython: exit 0,
# silent. So the rejection is independent of element count, non-emptiness, and
# whether the receiver is a bound name. ROOT CAUSE
# (src/lython/runtime/modules/builtins.mlir:17499): the native contract is
# declared func.func @LyList_ExtendM(%self: memref<9xi64> {ly.ownership.ob
#
# axes: argument container type: list AGREES; tuple, set, dict, str, and range
# each fail with the same shape of error naming their own type ('cannot adapt
# builtins.set to runtime input 1 of builtins.list.extend [values:
# memref<11xi64>, expected memref<9xi64>]', likewise builtins.dict
# memref<8xi64>, builtins.str, builtins.range memref<5xi64>). So the parameter
# is bound to the physical list layout rather than an iterable protocol.
# literal vs variable: `pairs: list[tuple[str,int]] = [...]` then extend still
# fails for the non-list types, and a bare tuple literal `xs.extend(("z",))`
# fails too -- not a literal-inference issue. wrap-in-list:
# `xs.extend(list(d))` AGREES and `xs.extend([v for v in range(3)])

# ==========================================================================
# [MISSING-FEATURE] functions
# *args and **kwargs on a method are never bound: the parameter name is
# unresolved and its static type is None
#
#     --- program
#     class Registry:
#         def many(self, *items: str) -> int:
#             return len(items)
#     
#     
#     print(Registry().many("p", "q"))
#
# lyc: fail.py:6:6: emit error: too many positional arguments for inlined
# class method fail.py:3:19: emit error: unresolved name 'items' fail.py:3:15:
# emit error: static type !py.literal<None> does not provide manifest method
# '__len__' [exit 1] (byte-identical to the claim except the filename; I wrote
# the program myself)
# py : 2 [exit 0]
#
#     --- a neighbour that AGREES
#     def many(*items: str) -> int:
#         return len(items)
#     
#     
#     print(many("p", "q"))
#     
#     # both compilers print 2, exit 0
#
# verifier: VERDICT: real, correctly described, not excluded. The finder's
# diagnostic text is exact and its neighbour is exact. Files (all mine, under
# .../scratchpad/wf/verify_dom_functions_2/): fail.py, neigh.py, min.py,
# b_zero.py, c_one.py, d_kwargs.py, e_static.py, f_unbound.py, h_mixed.py,
# i_init.py, j_ignore.py, n1_modfunc.py, n2_modkw.py, n4_plain.py,
# n5_nestfn.py, g_nestplain.py SMALLER REPRODUCER (min.py) — drops len(),
# drops the misleading arity error, 4 lines: class C: def m(self, *a: int) ->
# None: print(a) C().m() lyc: "min.py:3:14: emit error: unresolved name 'a'"
# [exit 1] cpython: "()" [exit 0] Calling with zero variadic args (b_zero.py)
# is the cleaner statement of the bug: the "too many po
#
# axes: free function vs method: identical signature as a free function works.
# Method with a normal list parameter (def many(self, items: list[str]))
# works. Zero-argument call Registry().many() STILL fails with the same
# 'unresolved name items' plus the None/__len__ cascade, so the body is
# rejected regardless of the call site. **kwargs on a method fails the same
# way plus 'unexpected keyword argument a/b for inlined class method'.
# @staticmethod with *args fails identically. The three diagnostics are a
# cascade from one cause: the vararg parameter is not entered into the
# method's scope at all, so it resolves to the None literal.

# ==========================================================================
# [FIXED 2026-08-19] generators + ownership   FOUND the same day, batch of 20
#                                             realistic programs
# A generator that iterates an ITERATOR OBJECT (range, or a list) and has any
# branch containing a may-raise call is refused: the iterator "is still owned
# when a call may unwind out of the function"
#
#     --- program
#     def evens(n: int):
#         for i in range(n):
#             if i % 2 == 0:
#                 yield i
#
#     print(list(evens(6)))
#
# lyc: exit 1, loc(...def...): error: owned resource from @LyRange_Iter result 0
# is still owned when a call to 'LyLong_FromI64' may unwind out of the function;
# the unwind path must release, transfer, or return it
# py : [0, 2, 4]
#
#     --- a neighbour that AGREES
#     def evens(n: int):
#         i = 0
#         while i < n:
#             if i % 2 == 0:
#                 yield i
#             i += 1
#
# axes, all measured:
#   - no branch at all (`for i in range(n): yield i`) -> AGREES. The branch is
#     required.
#   - the branch need not contain the YIELD: a branch containing `print(x)`
#     with the yield after it is refused too (v1), and `yield` then a branch
#     with a print is refused with a DIFFERENT producer
#     (@__ly_generator_claim_builtins_range_iterator, v2), so the token is the
#     frame's claim on the iterator either way.
#   - `continue` inside the branch instead of a yield -> refused (v3).
#   - `for i in list(range(n))` -> refused, producer @LyList_FromLength (w2).
#     `it = range(n); for i in it` -> refused (w1). So it is the iterator
#     OBJECT, not range specifically.
#   - `for i in xs` over a list PARAMETER -> AGREES (u2): that walk is
#     evidence-backed and claims no iterator object.
#   - a plain (non-generator) function with the same loop and branch -> AGREES
#     (u3). The generator's resume clone is where it happens.
#   - `for i in range(len(xs))` with `xs[i]` in the CONDITION -> AGREES (v5),
#     which is the one axis I cannot explain and the one a repair should look
#     at first.
#   - a while loop with the same branch -> AGREES (w3), which is the workaround.
#
# ⭐ LOCALISED EXACTLY 2026-08-19, with LYTHON_TRACE_UNWIND_HOLD (added in the
# same session: the insertion pass prints its per-(call, group) verdict and the
# verifier prints the group it refuses on, so the pair names the disagreement).
#
# THE VERIFIER IS RIGHT, and that is the first thing to know. The call it fires
# on is the LyLong_FromI64 in the SUSPEND block -- the one that boxes the yielded
# value just before the resume clone returns. The iterator is transferred out in
# that return's suspend lanes, so an unwind out of the boxing call leaves the
# frame with nobody holding it: a real leak on the exception path, not a false
# refusal about one.
#
# WHY NO CLEANUP IS PLACED. At that call NO tracked group answers Held. The
# producer group (@LyRange_Iter result 0) answers UNKNOWN -- its token moved into
# the loop's block argument, and the dead-set walk can only say "some paths" --
# and the block argument that actually carries the iterator INTO the suspend
# block (^bb13's third argument, the one the return forwards) is not among the
# tracked groups at all. The pass skips Unknown on purpose: releasing a token it
# cannot prove held is a double free, which is worse than the leak.
#
# THE MISSING GROUP, narrowed further the same evening with the root pointers in
# the trace: ^bb13's argument -- the one the suspend block's return transfers out
# -- HAS NO TRACKED GROUP AT ALL. The groups that exist are the loop header's and
# ^bb7's; the unwind pass gets its block-argument groups from
# insertOwnedBlockArgumentReleases run analysis-only, and that analysis yields
# the arguments that need a NORMAL-PATH release. An argument transferred out by a
# return needs none, so it is never collected -- and it is exactly the one an
# unwind has to release.
#
# ⛔ TWO REPAIRS MEASURED AND REVERTED IN groupTokenAtPoint (the "edge into the
# point's block" arm):
#   1. Reading such an edge as "delivered, not killed" fixes all six generator
#      programs AND turns two goldens into DOUBLE FREES
#      (except_handler_rebind_carry, method_return_through_try): there the edge
#      consumes the group and the block's argument belongs to a different one.
#   2. Narrowing it to "the receiving argument is IN this group" keeps 779/779
#      green and never fires for the generator, because of the missing group
#      above. Sound, useless, not shipped.
# So the repair really is the collection, not the walk.
#
# ⭐⭐ FIXED, and the fix is where the last line of this entry said it would be.
# `forwardedBlockArgGroup` bails when a terminator hands the same values to TWO
# successors ("group split across successors"), and a suspend is exactly that:
# `cond_br %susp, ^suspend(%it), ^loop(%it)`. The release side is right to bail --
# which destination would own it? -- but on an UNWIND both destinations hold the
# token, so the unwind pass now follows each successor itself and adds the
# destination groups to its own list only. Six programs that were refused now
# agree with CPython, including one that raises out of a suspended generator, and
# every one measures net 0 allocations. Pinned by
# tests/golden/cases/generator_over_range_with_a_branch.py.
#
# ⛔ Unwind-only on purpose: feeding these groups back to
# insertOwnedBlockArgumentReleases would place a NORMAL-path release for an
# argument another edge still carries.
#
# The trail below is kept because it is four rounds of localisation and the two
# reverted repairs are the reason the third one was written where it was.
#
# ⭐ AND THE COLLECTION'S OWN REASON, one level further: every candidate DOES
# become an unwind group (insertOwnedBlockArgumentReleases pushes each one), so
# what is missing is the CANDIDATE. Candidates are seeded from a group's
# forwarding terminators, and the seed here is the LyRange_Iter result -- whose
# forwarding edge reaches the loop header's argument, not the suspend block's.
# The chain stops one hop short: the loop header's argument is never re-seeded,
# so nothing forwards it to ^bb13. Iterating the seeding to a fixpoint is the
# repair; doing it for the RELEASE side too would place new releases, so the
# extra candidates belong to the unwind list alone.
#
# ⛔ Still not attempted: this is the generator frame, and
# [[lython-fragile-invariants]] says to write the invariant down before touching
# it. What to write first: which values the resume clone owns at each suspend
# point, and which of them the return transfers.

# ==========================================================================
# [GAP BATCH] shipped stdlib, ten realistic programs   FOUND 2026-08-19
# Five of ten programs that use a shipped module do not compile. Each is
# reduced, and each is a different mechanism -- the point of listing them
# together is that none of the five is about the module it appears in.
#
# 1. json.dumps of a CONCRETE dict.  `json.dumps({"a": 1})` ->
#    "!py.callable<[json.JSONValue, ...]> is not callable: call arguments do
#    not match the Callable contract". dumps takes JSONValue (the recursive
#    union), and dict[str, int] is not dict[str, JSONValue] -- a mutable
#    container is invariant, and CPython's own type checkers say the same
#    thing about it (typeshed declares obj: Any). What would make the call
#    work is BUILDING a union tree at run time, which is the union mechanism
#    already recorded above. json.loads and its subscripts work.
#
# 2. collections.defaultdict and collections.deque are simply absent, and
#    collections.py says why in its docstring: CPython implements both in C,
#    so the layering rule puts them in runtime/modules/*.mlir, and no
#    _collections manifest exists. That is a new native container each --
#    header, allocator, deallocator, methods -- not a lib/*.py addition.
#
# 3. itertools.islice over a non-indexable: "islice() as a value requires
#    indexable sequences (list/str/tuple/bytes); iterate non-indexable sources
#    directly". `itertools.islice(itertools.count(5), 3)` is the canonical use
#    (bounding an infinite iterator) and is exactly what it refuses.
#
# 4. functools.lru_cache as a decorator: "decorator 'functools.lru_cache' is
#    not supported (unrecognized decorators are rejected instead of silently
#    ignored)". The refusal is deliberate and the message is right; what is
#    missing is the feature.
#
# 5. `import os.path` -- FIXED the same day, see wb_grid_leftovers (51).
#
# The five that pass are worth naming too, because they are the ones a reader
# would expect to break first: dataclasses with a default_factory field, an
# Enum iterated and looked up by value, string.maketrans/translate, bisect
# insort, and io.StringIO.

# ==========================================================================
# [GAP] generators, two that survive the split-forward fix   FOUND 2026-08-19
# Four more generator shapes were run once the range+branch refusal was fixed.
# Two work and are worth naming (a nested double loop yielding a tuple, and an
# early `return` inside the loop); two do not, and neither is the ownership
# path:
#
# 1. YIELDING A DICT KEY. `for k in d: yield k` inside a generator ->
#    "source generator next lowering currently supports yields whose runtime
#    value is a single lane, and '!py.contract<"builtins.str">' has 2".
#    Yielding a str is NOT the problem -- `yield "a"` works, and so does
#    `for s in xs: yield s` over a list[str], with or without a branch.
#
#    ⭐ WHY, read out of the two lowerings: the STATE MACHINE
#    (GeneratorStateMachine.cpp) carries lane GROUPS and would take this, but
#    its eligibility scan rejects a body containing any op with regions except
#    py.try -- and the dict walk is one. The generator then falls to the inline
#    path in SourceGenerator.cpp, whose SourceYieldPlan holds ONE SSA value per
#    yield, and a str is two. So the repair is either to let the state machine
#    accept the dict walk's region op or to give the inline path a lane group;
#    the note above that refusal already says the second is unbuilt.
#
# 2. FIXED THE SAME DAY -- A REBOUND EMPTY LIST ACROSS A YIELD.
#        buf: list[int] = []
#        buf.append(1)
#        yield buf
#        buf = []          <- list[object] here
#        yield buf
#    -> "runtime bundle for '!py.union<list[int], list[object]>' has 1 values".
#    The empty-literal rule the emitter applies outside a generator -- an empty
#    literal has no element type of its own, so a rebind with one keeps the type
#    the name already has -- was missing from the generator's frame analysis,
#    which overwrote the slot with list[object]. Fixed in
#    bindGeneratorAnalysisTarget; pinned by `refilled` in
#    tests/golden/cases/generator_over_range_with_a_branch.py.
#
#    ⛔ The same idiom INSIDE A LOOP (append, yield when full, rebind, keep
#    going) is still refused, now for an OWNERSHIP reason: "owned resource from
#    @LyList_FromLength result 0 is still owned when a call may unwind". Seeding
#    the split-forward chain from every tracked group as well as from the
#    block-argument ones was measured on exactly this program and changed
#    nothing, so the chain is not what it needs.

# ==========================================================================
# [FIXED 2026-08-19] `print([A()])` for a class with no __repr__
# The container element repr ASSERTED rather than falling back, so a plain class
# in a list aborted where CPython prints `[<__main__.A object at 0x...>]`. The
# fallback exists now (the default repr, which names the real class), and what
# stood in the way was NOT the repr.
#
# ⭐ THE CAUSE, found by reading the emitted LLVM IR of the fallback helper: a
# hand-written manifest helper that returns an OWNED result but carries no
# `ly.runtime.*` attribute is NOT a manifest function to
# `own::isRuntimeManifestFunction`, so the refcount pass treats it as USER code
# and inserts a release for the hook result on the path that does not use it.
# The hook's miss returns `ub.poison`, so that release freed garbage --
# `call void @LyUnicode_DecRef(...)` right at the top of the fallback block --
# and the program aborted inside malloc. Adding the attributes fixes it; the
# assert had been standing between that call and the corruption.
#
# The four measurements that got there, kept because each one ruled something
# out: calling the hook and IGNORING its results still crashed (so it was not
# the results), not calling it at all printed and exited 0, a class with fields
# crashed the same way (not the field layout), and the abort came from
# libsystem_malloc at `-jit-codegen-opt=none` as well as at the default (so not
# the optimizer exploiting poison).
#
# ⛔ SIXTY-FOUR OTHER HELPERS have the same shape -- owned results, no
# ly.runtime.* attribute -- across builtins/_io/asyncio/lyrt/unicodedata. They
# work today because they use their results on every path, but each is one
# unused path away from the same corruption, and marking them all is NOT a safe
# sweep: some may rely on the pass's insertion, so removing it would leak. The
# scan is one `grep` (owned_results without ly.runtime.*), and a round that
# wants to close this should measure each one rather than annotate them all.

# ==========================================================================
# [GAP] a user-defined decorator                             FOUND 2026-08-19
# The plain decorator idiom is refused twice over, and the two halves are
# independent:
#
#     def twice(fn):                    # 1. parameter 'fn' requires an
#         def wrapper(n: int) -> int:   #    annotation
#             return fn(fn(n))
#         return wrapper
#
#     @twice                            # 2. decorator 'twice' is not supported
#     def inc(n: int) -> int:           #    (unrecognized decorators are
#         return n + 1                  #    rejected instead of ignored)
#
# ⭐ HALF ONE IS A ONE-PLACE FIX AND WAS MEASURED. An unannotated parameter IS
# inferred from call sites -- `def g(x): return x + 1` with `g(1)` works, and so
# does passing a FUNCTION (`apply(fn, n)` called as `apply(inc, 1)`) -- but
# `collectModuleCallNodes` returns at a FunctionDef, so a decorator application
# is never collected as a call site. Synthesizing `twice(inc)` there removes the
# annotation error entirely and leaves only half two. Not shipped, because with
# half two still refusing there is nothing a test can see.
#
# ⛔ HALF TWO IS A FEATURE: checkDecorators keeps a whitelist (staticmethod,
# classmethod, property, abstractmethod, dataclass, native, typing markers), and
# anything else is rejected on purpose. Supporting user decorators means
# desugaring `@deco def f(...)` into `f = deco(f)`: emit the function, take its
# function OBJECT (emitFunctionObject already exists, and indirect calls work --
# see `apply` above), call the decorator, and bind the name to the result, which
# makes every later call of that name indirect.
#
# ⛔ The decorator FACTORY shape (`@register("a")`) needs one more thing: the
# inner `deco(fn)` is a NESTED function, which the module-level parameter
# fixpoint does not cover at all.

# ==========================================================================
# [BUG] a closure that captures a FUNCTION            FOUND + FIXED 2026-08-19
# Found while scoping the decorator feature above, and it is the third half of
# it: what `@deco` desugars to is exactly this program, written by hand.
#
#     def wrap(fn: Callable[[int], int]) -> Callable[[int], int]:
#         def inner(n: int) -> int:
#             return fn(n)
#         return inner
#
#     # function target wrap$inner$1$5_4 closure 0 has contract
#     # '!py.contract<"builtins.function">', expected
#     # '!py.callable<[int], returns=[int]>'
#
# ⭐ THE ASYMMETRY IS THE WHOLE DIAGNOSIS: capturing an INT worked (`adder(10)`
# returning an inner that adds k), and CALLING a callable-typed parameter worked
# (`apply(fn, n)`). Only the two together failed. A function value has one
# physical shape in this ABI, `builtins.function` -- which is why the call
# through a parameter works at all -- while the closure SLOT's declared type is
# the emitter's precise callable, taken from the annotation. FunctionTargetCalls
# compared the erased value against the precise slot with isAssignableTo and
# refused. Relaxed in exactly that one direction, with the comment saying why.
#
# ⛔ NOT A HOLE IN THE CHECKING, and this was measured both ways: `wrap(shout)`
# with `shout: Callable[[str], str]` is still refused, at the EMITTER, with
# "call arguments do not match the Callable contract" -- and so is an inner that
# calls the captured fn with the wrong arity. The lowering was re-asking a
# question the emitter had already answered, at a point where the answer had
# been erased; only that re-ask is relaxed.
#
# Pinned by tests/golden/cases/closure_captures_a_function.py -- twice() calls
# the captured function twice and compose() threads two of them, which
# distinguishes "the right function was captured" from "a function was".
#
# ⛔ The decorator SYNTAX is still refused (half two above). This closes the
# thing that would have made the desugar produce a program that does not
# compile, so that entry now has one fewer unknown rather than one fewer half.

# ==========================================================================
# [GAP x2] exceptions, from a four-program probe      FOUND + 1/2 FIXED 2026-08-19
# Two of four ordinary exception programs do not compile; the other two (a
# two-argument ValueError read through e.args, and a user class calling
# super().__init__ with an f-string) do.
#
# 1. FIXED 2026-08-19 (night). A NON-STR EXCEPTION ARGUMENT. `raise
#    ValueError(42)` -> "cannot adapt builtins.int to runtime input 3 of
#    builtins.ValueError.__init__".
#
#    ⭐ THE ASYMMETRY NAMED THE MECHANISM: `raise ValueError("m", 2)` -- strictly
#    more work -- already compiled. TWO arguments go into the payload block
#    (boxed, .args reads them back); ONE goes into the message LANE, which is a
#    unicode, so anything else had nowhere to go. One non-str argument now takes
#    the same block. The renderer had to learn CPython's one-argument case with
#    it: str(e) is str(args[0]) for one and "(a, b)" for two-and-up, which is
#    why `str(ValueError(42))` is "42" and not "(42,)".
#
#    ⛔ KeyError.__str__ IS repr(args[0]) and it is INHERITED, so the renderer
#    asks the class taxonomy (LyEH_ClassIdMatches) rather than comparing one
#    class id. The str path kept that override inside LyKeyError_Init; routing
#    the non-str argument through the generic block would have lost it, and
#    `str(KeyError(p))` printed p's __str__ where CPython prints its __repr__.
#    Both halves are in the golden, with a Point whose two dunders differ.
#
#    ⛔ SystemExit IS HELD BACK on purpose, and the two defects that hold it are
#    recorded below. Pinned by
#    tests/golden/cases/exception_argument_is_not_a_string.py.
#
# 2. `type(e.__cause__).__name__` -> "type(x) needs a statically resolved class,
#    and !py.union<...> is not one". __cause__ is `BaseException | None`, and the
#    class-name read added today answers a single contract. A union could be
#    answered per tag (the value carries one), which is the same shape as every
#    other union read and waits on the same mechanism.

# ==========================================================================
# [BUG] an exception argument that outlives the raise  FOUND + FIXED 2026-08-19
# Found while leak-gating the non-str argument fix, on the loop the gate itself
# wanted, and it predates that fix -- the two-argument form has always gone
# through the same lowering:
#
#     i = 0
#     while i < 5:
#         try:
#             raise ValueError(i)     # or ValueError(i, 0), which is older
#         except ValueError:
#             pass
#         i += 1
#     # owned resource from @LyLong_FromI64 result 0 is released or
#     # transferred more than once on one CFG path
#
# ⭐ IT IS A DOUBLE FREE, not a strict verifier: with --release it aborts. The
# payload block retains its own reference to each argument AND released the
# argument's. That pair is exactly right for a TEMPORARY -- one token, nobody
# else to discharge it -- and wrong for anything that outlives the raise: `i +=
# 1` releases the old int and the loop's own token releases it again.
#
# ⛔ THE PREDICATE IS "does this value already have other users", asked BEFORE
# the loop emits anything, because afterwards every argument has the retain and
# the store among its users. It is the same question the sequence-literal path
# asks with valueIsConsumedOnlyBy; that path has the py-level operand to ask it
# of, and this one has the lowered handle, which is why the spelling differs.
#
# ⛔ DROPPING THE RELEASE ALONE MADE IT WORSE, and the failure named the missing
# half: `break` inside the handler turned into "ownership CFG exploration
# exceeded 20000 states (last: retained=1000)". An aggregate retain with no
# parent and no local release is a token the walk carries forward, one per
# iteration. chargeSlotRetainsToParent says the retain belongs to the EXCEPTION
# -- the collection paths already call it, and this one never had to, because
# the release it no longer emits used to cancel the retain on the spot.
#
# ⛔ WHY IT LOOKED LIKE A LOOP BUG: `raise ValueError(i)` outside a loop is fine
# (the value dies right after), and `for j in range(5)` is fine (j is not
# rebound by a store that releases the old value). It needs a name that is BOTH
# reassigned and passed, which is why `while i < n: ... raise E(i) ... i += 1`
# is the shape and `while n < 5: raise E(i)` is not.
#
# Pinned by tests/golden/cases/an_exception_argument_that_outlives_the_raise.py,
# which reads the argument back out of e.args so a value freed under the block
# is a read-after-free rather than a leak.

# ==========================================================================
# [BUG x2] SystemExit's code, both directions          FOUND + FIXED 2026-08-19
# Found by asking what the non-str exception argument fix above would do to
# SystemExit, and the answer was "mis-execute", so it was excluded from it for
# one commit and then repaired. Two defects, one mechanism: the top-level runner
# read "the message is empty" as "use the status LyHost_SetExitStatus recorded",
# which is a PROXY for "this came from sys.exit" and got both edges wrong.
#
# 1. `raise SystemExit(3)` -> refused ("cannot adapt builtins.int to runtime
#    input 3"). CPython exits 3, silently.
#
# 2. `raise SystemExit("")` -> exited 0, silently. CPython prints an empty line
#    to stderr and exits 1. An empty message is indistinguishable from
#    `SystemExit()` (which IS exit 0) under the proxy, and this half needed no
#    new feature to be wrong -- it was wrong on its own.
#
# ⭐ THE FIX IS ONE CHANGE OF SIGNAL, not two repairs. The exception object grew
# a sixth word for the code (biased by one, so slot 0 is "no int code" rather
# than "exit 0"), SystemExit's argument goes into the payload block whether or
# not it is a str -- which is what makes `SystemExit()` and `SystemExit("")`
# different shapes -- and the runner asks those two words instead of the message
# length: a code exits with it in silence, no argument at all exits 0 in
# silence, anything else prints and exits 1. `g_sys_exit_status` and
# `LyHost_SetExitStatus` are gone; the status rides the exception now, which is
# where CPython's .code has always been, so two SystemExits in flight can each
# carry their own.
#
# ⛔ THE CODE IS RECORDED AT CONSTRUCTION, not read back at the raise, because
# the block holds the argument BOXED and pulling an i64 out of a box needs a
# per-class unboxer the runner cannot call. bool gets its own entry point for
# the same reason (one i1 lane, not the int triple) -- and it needs one, because
# CPython counts a bool as an int: `raise SystemExit(True)` exits 1 in silence.
#
# ⛔ The stale rationale was written down in sys.mlir: "an int code must go
# through sys.exit (the exception object has no code slot)". That was true of
# the 5-word object and stopped being interesting the moment the payload block
# existed; the slot cost one word.
#
# ⛔ WHAT IS STILL DEVIATION: `sys.exit(7)` sets the status but not .args, so a
# CAUGHT one has str(e) == "" and e.args == () where CPython gives "7" and (7,).
# Boxing 7 into the block needs the 16-word box layout the LOWERING computes,
# and no manifest-side store reaches it. Pinned as "was it caught" rather than
# "what does it carry" in system_exit_is_an_exception.py, with the reason.
#
# Pinned by tests/golden/errors/system_exit_int_code.py (exit 3),
# tests/golden/errors/system_exit_empty_message.py (exit 1, and RED at 0 on the
# pre-fix binary with no other change), and
# tests/golden/cases/system_exit_is_an_exception.py.

# ==========================================================================
# [GAP] a recursive generator, in three stages       FOUND + 2/3 FIXED 2026-08-19
# Every tree walk is written as a recursive generator, and this one failed at
# three different stages, each hidden behind the one before it:
#
#     def walk(n: Node) -> Iterator[int]:
#         yield n.v
#         for k in n.kids:
#             for v in walk(k):
#                 yield v
#
# 1. FIXED. The generator ANALYSIS could not type the self-call, so the yield
#    type came out `object` and an annotated generator was refused as "annotated
#    Iterator[int] but yields object". The annotations are right there: the
#    function's own name is now bound, for that walk, to a callable built from
#    them. Built from the annotations rather than from functionSignature because
#    that call is what is running and its memo is not filled yet -- asking it
#    recurses forever.
#
# 2. FIXED. The EMITTER bound the name inside the body to `sig.callable`, which
#    for a generator is the BODY's signature and returns None -- so the self-call
#    typed as None and the refusal read "literal<None> does not provide manifest
#    method '__iter__'", which names nothing the reader wrote. A generator's own
#    name denotes a GENERATOR: `publicCallable`. Pinned by
#    EmitterTest.ARecursiveGeneratorTypesAndLeavesTheRefusalToTheLowering.
#
# 3. NOT FIXED, and now visible: "yield from delegation exceeded the static
#    inlining budget (recursive delegation has no static expansion)". Delegation
#    is expanded by INLINING the delegate, and a self-call has no static
#    expansion -- this is the nested-generator frame, the same mechanism the
#    `for x in G(): yield x` rewrite (wb_grid_leftovers (42)) documents. Every
#    materialising workaround inside a generator (`list(count(n-1))`,
#    `sum(count(n-1))`) hits the neighbouring wall: "a generator returned out of
#    a function cannot be resumed".

