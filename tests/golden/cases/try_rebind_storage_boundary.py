# The boundary of the try storage promotion, pinned so the next change to
# `emitTry` does not widen it back.
#
# A local the try body rebinds through a NAME target is promoted to a cell for
# the extent of the statement, which is what makes the handler and the
# continuation see the value at the raise point (cases/try_handler_entry_binding).
# Three kinds of local must stay OUT of that promotion; each is pinned below.
#
#  1. A structural-mutation receiver (`xs.append(v)` / `d[k] = v` / `d |= o`).
#     Its evidence lives on the SSA value, so a cell load would demote the
#     receiver, and the mutation rebinds the binding directly, which would
#     bypass the cell the continuation reads. Inside a try those shapes are
#     still loudly rejected (errors/try_structural_rebind,
#     errors/try_dict_merge_rebind); what is pinned here is that a receiver
#     mutated OUTSIDE the try keeps working next to a promoted local.
#  2. A name first bound inside the try body: there is no pre-try incarnation
#     to merge, so it travels the post-try result lanes.
#  3. A loop-carried local: moving a loop block argument's token into an
#     aggregate slot inside the same iteration is mis-tracked by release
#     insertion, so it keeps the result lanes too.


def promoted_next_to_a_mutated_receiver() -> str:
    label = "before"
    xs = [1]
    xs.append(2)
    try:
        label = "inside"
        raise ValueError()
    except ValueError:
        pass
    xs.append(3)
    return label + str(len(xs))


print(promoted_next_to_a_mutated_receiver())


def bound_inside_only() -> int:
    try:
        fresh = 5
    except ValueError:
        fresh = 6
    return fresh


print(bound_inside_only())


def loop_carried() -> str:
    acc = "a"
    for i in range(3):
        try:
            if i == 1:
                raise ValueError()
            acc = acc + str(i)
        except ValueError:
            acc = acc + "-e"
    return acc


print(loop_carried())
