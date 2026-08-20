# Findings from the session that gave a base-typed receiver a dispatch.
# Every program here was run under both `./build/bin/lyc jit` and
# `python3.14`, on the same day, and reduced until one axis moved the outcome.
#
# What SHIPPED (so a reader does not re-find it):
#   - a base-typed receiver reaches the overriding body, through a synthesized
#     module function per (class, method) that tests the runtime class;
#   - the operator spellings (len/str/repr/print/==/+/[]) take the same
#     dispatcher;
#   - a class declared further down the module is still a subclass, for both
#     `isinstance` and the narrowing it feeds.
#
# ==========================================================================
# [GAP] a subclass field that is not an int cannot be seen through the base
#
#     class A:
#         def __init__(self, s: str) -> None:
#             self.s = s
#
#     class B(A):
#         def __init__(self, w: float) -> None:
#             super().__init__("b")
#             self.w = w
#
#     def show(a: A) -> float:
#         if isinstance(a, B):
#             return a.w        # <-- here
#         return 0.0
#
#     print(show(B(1.5)), show(A("x")))
#
# lyc: error: field 'w' of class B is not carried by a value typed as its base:
#      an instance is passed as its fields, and only int and bool fields are
#      stored in the instance itself, so a wider reference has no lane for this
#      one   (exit 1; before 2026-08-21 this read "class field ABI exceeds
#      object payload", which named neither the field nor the reason)
# py : 1.5 0.0
#
# axes, measured the same day:
#   extra field `m: int`            -> COMPILES and agrees. int and bool are
#                                      stored IN the instance
#                                      (`classFieldStoredBoxed`), so they need
#                                      no payload word of their own.
#   extra field `w: float`          -> aborts, reading it or not: with the read
#                                      removed the message becomes "cannot pass
#                                      concrete object B as builtins.object",
#                                      so the loss is at the ARGUMENT boundary,
#                                      not at the read.
#   extra field `l: N` (a class)    -> aborts the same way. This is why a
#                                      RECURSIVE class (`Branch(left: Node,
#                                      right: Node)`) cannot be dispatched
#                                      today, which is most tree code.
#   base with NO fields, subclass with two floats, through `list[A]`
#                                   -> COMPILES and agrees. A list element is
#                                      boxed, so the box carries the subclass's
#                                      words.
#
# So the boundary is the base-typed VALUE's lane count: an instance is lanes,
# not a pointer, and a base-typed lane list has room for the base's fields
# only. `classFieldValueOffset` computes the subclass's offset from the
# subclass's own field list and the read runs off the end of what the base
# carried -- which the lowering catches (AttributeOps.cpp), so this is a
# refusal and not a wrong answer.
#
# What it would take: size a class's payload for the widest class in its
# subtree, so a base-typed value can hold any subclass instance. Single
# inheritance appends fields, so the subclass's offsets stay valid; the cost is
# lanes on every base-typed value, and the change reaches construction, calls
# and field access together.
#
# ==========================================================================
# [GAP] a list literal of two subclasses does not become list[Base]
#
#     xs = [A(), B()]           # B(A)
#     for x in xs:
#         print(x.t())
#
# lyc: emit error: static type !py.union<!py.contract<"A">, !py.contract<"B">>
#      does not provide manifest method 't'
# py : A B
#
#     def join(xs: list[A]) -> str: ...
#     join([A(), B()])          # same union, refused at the call
#     join([B(), B()])          # list[B] against list[A], refused at lowering
#
# The ANNOTATED spellings work: `xs: list[A] = [A(), B()]` and
# `xs: list[A] = [B(), B()]` both compile and agree, so the machinery is the
# expectation, not the coercion. `emitCallOperands` already distributes an
# expected type per positional argument -- but only when the callee's contract
# reaches it, which the module-function path does not do.


# ==========================================================================
# [CRASH] a class named like an ALWAYS-LINKED manifest class  PARTLY FIXED
#
#     class Task:
#         def __init__(self, name: str, pri: int) -> None: ...
#     def top(tasks: list[Task], n: int) -> list[str]: ...
#     top([Task("a", 1)], 1)
#
# Two separate faults, one behind the other:
#
# 1. FIXED. `contractAnnotationName` claimed five BARE spellings for manifest
#    contracts whether or not anything imported them -- Task, Future,
#    AbstractEventLoop, CancelledError, Context -- so the parameter typed as
#    asyncio's Task and the call was refused ("arguments do not match Callable
#    contract for function target top"). A class the program declares now wins
#    for a bare name.
#
# 2. FIXED. With the annotation right, the LOWERING then gave the instances
#    asyncio's class id 15: `classContractCandidates` guesses `builtins.`,
#    `types.`, `_asyncio.`, `asyncio.` and `contextlib.` in front of a bare
#    class name, which is how a manifest `py.class @Task` finds `_asyncio.Task`
#    -- and a source class has no marker to tell it apart. rc=139. Source class
#    ops now carry `ly.class.source` and skip the guess.
#
# 3. NOT FIXED, same shape, two names: `class TaskIter` and `class FutureIter`
#    still rc=139 on
#
#        class TaskIter:
#            def __init__(self, v: int) -> None: self.v = v
#        def take(xs: list[TaskIter]) -> int: return xs[0].v
#        print(take([TaskIter(3)]))
#
#    while `TaskIter(3).v` on its own is fine, so it is the list/parameter
#    path. Neither the class id (the guess is skipped now) nor the manifest
#    shape lookup (keyed by the full contract) explains it, and the class ops
#    are gone by the runtime-import dump, so the remaining reader is somewhere
#    that still maps a bare name onto `_asyncio.*`. Worth an hour when someone
#    is next in that file; nobody names a class TaskIter, which is why it is
#    recorded rather than chased now.
#
# ⛔ And the PROTOCOL spellings (Sequence, Iterator, Generator, eleven more)
# keep the old refusal on purpose: letting a declared class win over those was
# built and reverted the same hour, because the emitter's own iteration typing
# asks for `Iterator` by name, so `class Iterator` broke every `for` loop in
# the program with "protocol Iterator does not provide manifest method
# '__next__'". Fixing it means giving the compiler's internal spellings a form
# a program cannot shadow.
