# contextlib.nullcontext.__enter__ returned the context manager itself, where
# CPython returns self.enter_result -- None by default. `with nullcontext() as
# c` was binding the wrong object. Asserted through `is None` rather than
# print(c) because printing a None-typed value is a separate, unrelated gap
# ("types.NoneType runtime object has no physical header value", which
# `x = None; print(x)` hits too).
from contextlib import nullcontext

with nullcontext() as c:
    print(c is None)

# The no-binding form and the body's normal completion still work.
with nullcontext():
    print("body")

# An exception propagates: __exit__ returns False, so it is not suppressed.
try:
    with nullcontext():
        raise ValueError("through")
except ValueError as e:
    print(str(e))

# Nesting, and a binding that outlives the block.
with nullcontext() as outer:
    with nullcontext() as inner:
        print(outer is None, inner is None)
print(outer is None)
