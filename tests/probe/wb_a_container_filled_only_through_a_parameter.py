# OPEN. An empty container filled ONLY through a callee's parameter, where that
# parameter is itself inferred, is a circle the seed scan does not close:
#
#     def put(heap, value):
#         heap.append(value)
#     h = []
#     put(h, 5)
#     print(h[0] + 1)
#     # builtins.object does not provide manifest method '__add__'
#
# What `put` would say about `h` is exactly the answer being asked for: `heap`
# has no type except the one the call site gives it, and the call site is the
# container being decided.
#
# MEASURED 2026-09-09, RelWithDebInfo:
#
#   `def put(heap: "list[int]", value: int)` ................ CORRECT NOW
#   the same with `heap` and `value` unannotated ............ refused
#   the container a LOCAL rather than a global, same shape .. refused
#   `h.append(5)` written at the call site instead .......... correct
#
# ⭐ THE ANNOTATED HALF IS THE ONE WITH AN ANSWER TO GIVE. A callee that
# declares `list[int]` says the element as plainly as an append does, and the
# scan reads it now (cases/a_container_handed_to_a_declared_parameter). The
# unannotated half has nothing to read.
#
# ⛔ Closing it means propagating in the other direction as well -- from
# `heap.append(value)` and `value`'s own call sites back out to `h` -- which is
# an interprocedural pass over container element types rather than the forward
# scan this is. Scoped, not built.
#
# ⛔ And the failure direction is a refusal, not a wrong element: an erased
# container is refused wherever it is decoded.
def put(heap, value):
    heap.append(value)


h = []
put(h, 5)
print(h[0] + 1)
