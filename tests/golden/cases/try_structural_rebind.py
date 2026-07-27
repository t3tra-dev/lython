# A structural list mutation inside `try` keeps the mutation and reaches the
# print. The value is CPython's.
#
# **Moved from `errors/`, where it pinned a refusal.** The refusal was
# "released owned resource ... is used after release": the rebind family
# (`extend` consuming the receiver and handing back a fresh owned triple) had
# no unwind release plan for the POST-rebind token. The error stopped
# EXISTING rather than stopping being detected -- there is no post-rebind
# token: `builtins.list` is one handle, `LyList_ExtendM` is void, and it
# publishes through the handle the try block already names, so no second
# owned group is created on the mutating path for the unwind edge to plan for.
#
# Discriminators (rfc/lane-conversion-playbook.md section 6): run_case.py does
# not pass `--release`, so the verifiers are on.
#
# ⚠️ **The standing control named here is GONE, and saying so is the honest
# form.** It read "the standing control is the probe suite's
# `known_borrowed_set_add`, which stays LOUD because `set` still declares the
# transfer". Converting `builtins.set` and `builtins.frozenset` to one handle
# flipped that probe to OK and took `builtins.mlir` from 147 `transfer_args`
# declarations to 141. **No container contract declares one any more**, so
# there is no same-run control of this shape left to point at -- the remaining
# declarations are the exception family's `__init__` receivers plus the str and
# bytes box packers, none of which a container mutation reaches. Do not invent a
# replacement: the surviving discriminators are that the verifiers are on, that
# the mechanism is readable in the IR (`LyList_ExtendM` is void and publishes
# through the handle), and that the case is clean under libgmalloc with
# `MallocScribble`.
xs = [1, 2]
try:
    xs.extend([3, 4])
except ValueError:
    pass
print(xs)
