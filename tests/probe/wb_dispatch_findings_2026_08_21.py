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
# lyc: error: class field ABI exceeds object payload   (exit 1)
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
# carried -- which the lowering catches (AttributeOps.cpp, "class field ABI
# exceeds object payload"), so this is a refusal and not a wrong answer.
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
