# contextlib.nullcontext(value) enters as `value`, of value's own type; with
# no argument it is None. The manifest's nullcontext took no argument at all.
# WHY THIS IS RUN: the bound value is used as its own type in the body, and
# the exception has to leave the block unsuppressed.
from contextlib import nullcontext

with nullcontext(5) as v:
    print(v + 1)
with nullcontext("s") as s:
    print(s.upper())
with nullcontext() as n:
    print(n is None)
try:
    with nullcontext([1]) as xs:
        xs.append(2)
        raise ValueError(str(xs))
except ValueError as e:
    print(e)
